from pyexpat import features

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_

import numpy as np
from utils import misc
# from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
# from utils.logger import *
import random
from knn_cuda import KNN
from models.Modules import *
# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

### Mamba import start ###
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, PatchEmbed
from timm.models.vision_transformer import _load_weights

import math
import time
from collections import namedtuple

from .bimamba_ssm.modules.mamba_simple import Mamba
from .bimamba_ssm.utils.generation import GenerationMixin
from .bimamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

# from .rope import *
import random

try:
    from .bimamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

### Mamba import end ###

###ordering
import math
# from models.z_order import *


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def patch_random_walk(neighbors, n_faces, fps_centers, seq_len):
    batch_size = neighbors.shape[0]  # 假设batch size由neighbors的第一维表示
    device = neighbors.device
    seqs_arr = []

    for batch_idx in range(batch_size):
        seq_batch = []
        jumps_batch = []

        for f0 in fps_centers[batch_idx]:  # 对于每个批次中的起始点列表
            seq = torch.zeros((seq_len + 1,), dtype=torch.int32)
            jumps = torch.zeros((seq_len + 1,), dtype=torch.bool)
            visited = torch.zeros((n_faces + 1,), dtype=torch.bool)
            visited[-1] = True
            visited[f0] = True
            seq[0] = f0
            jumps[0] = True
            backward_steps = 1

            i = 1
            while i <= seq_len:
                current_node = seq[i - 1]
                this_nbrs = neighbors[batch_idx][current_node]
                nodes_to_consider = [n for n in this_nbrs if not visited[n]]

                if len(nodes_to_consider) > 0:
                    # 有未访问的相邻点，选择其中一个
                    to_add = torch.tensor(nodes_to_consider)[torch.randint(len(nodes_to_consider), (1,))].item()
                    jump = False
                    backward_steps = 1
                else:
                    # 没有未访问的相邻点，开始回溯
                    jump = True
                    found_new_path = False

                    # 从回溯步数逐步增大，直到找到一个未访问相邻点或完全随机跳转
                    while not found_new_path and i > backward_steps:
                        back_node = seq[i - backward_steps - 1]
                        back_nbrs = neighbors[batch_idx][back_node]
                        back_nodes_to_consider = [n for n in back_nbrs if not visited[n]]

                        if len(back_nodes_to_consider) > 0:
                            # 找到未访问的相邻点，替换回溯路径上的所有死路点
                            to_add = torch.tensor(back_nodes_to_consider)[
                                torch.randint(len(back_nodes_to_consider), (1,))].item()
                            for replace_idx in range(i - backward_steps, i):
                                seq[replace_idx] = to_add
                                visited[to_add] = True
                            found_new_path = True
                            i = i - backward_steps  # 回到新路径点重新开始
                            break
                        else:
                            backward_steps += 2

                    # 如果回溯没有找到未访问相邻点，则完全随机选择一个新点
                    if not found_new_path:
                        to_add = torch.randint(n_faces, (1,)).item()
                        backward_steps = 1

                visited[to_add] = True
                seq[i] = to_add
                jumps[i] = jump
                i += 1

            seq_batch.append(seq)
            jumps_batch.append(jumps)

        seq_batch = torch.stack(seq_batch)  # 将该批次的中心点序列堆叠
        seqs_arr.append(seq_batch)
    seqs_arr = torch.stack(seqs_arr)
    return seqs_arr.to(torch.int64).to(device)

class Encoder(nn.Module):  ## Embedding module
    def __init__(self, input_channel, encoder_channel):
        super().__init__()
        self.input_channel = input_channel
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(input_channel, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.input_channel)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1))  # BG 256 n
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1, -1, n), feature], dim=1)  # BG 512 n
        feature = self.second_conv(feature)  # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, ring_1, vertices, faces):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape  # B N 3
        # fps the centers out
        center, center_idx = misc.fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center)  # B G M : get M idx for every center
        idx = patch_random_walk(ring_1, xyz.shape[1], center_idx, seq_len=self.group_size-1)
        assert idx.size(1) == self.num_group  # G center
        assert idx.size(2) == self.group_size  # M knn group

        neighborhood = index_points(xyz, idx)
        neighborhood = neighborhood - center.unsqueeze(2)
        neighborhood_idx = idx

        # idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        # idx = idx + idx_base
        # idx = idx.view(-1)
        # neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        # neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # # normalize: relative distance
        # neighborhood = neighborhood - center.unsqueeze(2)
        # relative distance normalization : sigmoid
        # neighborhood = torch.sigmoid(neighborhood)
        return neighborhood, center, neighborhood_idx


class GroupFeature(nn.Module):  # FPS + KNN
    def __init__(self, group_size):
        super().__init__()
        self.group_size = group_size  # the first is the point itself
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz, feat):
        '''
            input:
                xyz: B N 3
                feat: B N C
            ---------------------------
            output:
                neighborhood: B N K 3
                feature: B N K C
        '''
        batch_size, num_points, _ = xyz.shape  # B N 3 : 1 128 3
        C = feat.shape[-1]

        center = xyz
        # knn to get the neighborhood
        _, idx = self.knn(xyz, xyz)  # B N K : get K idx for every center
        assert idx.size(1) == num_points  # N center
        assert idx.size(2) == self.group_size  # K knn group

        neighborhood = index_points(xyz, idx)
        neighborhood = neighborhood - center.unsqueeze(2)
        neighborhood_idx = idx
        neighborhood_feat = index_points(feat, idx)


        # idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        # idx = idx + idx_base
        # idx = idx.view(-1)
        # neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]  # B N K 3
        # neighborhood = neighborhood.view(batch_size, num_points, self.group_size, 3).contiguous()  # 1 128 8 3
        # neighborhood_feat = feat.contiguous().view(-1, C)[idx, :]  # BxNxK C 128x8 384   128*26*8
        # assert neighborhood_feat.shape[-1] == feat.shape[-1]
        # neighborhood_feat = neighborhood_feat.view(batch_size, num_points, self.group_size,
        #                                            feat.shape[-1]).contiguous()  # 1 128 8 384
        # # normalize
        # neighborhood = neighborhood - center.unsqueeze(2)

        return neighborhood, neighborhood_feat


class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

 
# Local Geometry Aggregation
class K_Norm(nn.Module):
    def __init__(self, out_dim, k_group_size, alpha, beta):
        super().__init__()
        self.group_feat = GroupFeature(k_group_size)
        self.affine_alpha_feat = nn.Parameter(torch.ones([1, 1, 1, out_dim]))
        self.affine_beta_feat = nn.Parameter(torch.zeros([1, 1, 1, out_dim]))

    def forward(self, lc_xyz, lc_x):
        # get knn xyz and feature
        knn_xyz, knn_x = self.group_feat(lc_xyz, lc_x)  # B G K 3, B G K C

        # Normalize x (features) and xyz (coordinates)
        mean_x = lc_x.unsqueeze(dim=-2)  # B G 1 C
        std_x = torch.std(knn_x - mean_x)

        mean_xyz = lc_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(knn_xyz - mean_xyz)  # B G 1 3

        knn_x = (knn_x - mean_x) / (std_x + 1e-5)
        knn_xyz = (knn_xyz - mean_xyz) / (std_xyz + 1e-5)  # B G K 3

        B, G, K, C = knn_x.shape

        # Feature Expansion
        knn_x = torch.cat([knn_x, lc_x.reshape(B, G, 1, -1).repeat(1, 1, K, 1)], dim=-1)  # B G K 2C

        # Affine
        knn_x = self.affine_alpha_feat * knn_x + self.affine_beta_feat

        # Geometry Extraction
        knn_x_w = knn_x.permute(0, 3, 1, 2)  # B 2C G K

        return knn_x_w


# Max Pooling
class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0]  # B 2C G K -> B 2C G
        return lc_x


# Pooling
class Pooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)[0]  # B 2C G K -> B 2C G
        return lc_x


# Pooling
class K_Pool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, knn_x_w):
        # Feature Aggregation (Pooling)
        e_x = torch.exp(knn_x_w)  # B 2C G K
        up = (knn_x_w * e_x).mean(-1)  # # B 2C G
        down = e_x.mean(-1)
        lc_x = torch.div(up, down)
        # lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1) # B 2C G K -> B 2C G
        return lc_x


# shared MLP
class Post_ShareMLP(nn.Module):
    def __init__(self, in_dim, out_dim, permute=True):
        super().__init__()
        self.share_mlp = torch.nn.Conv1d(in_dim, out_dim, 1)
        self.permute = permute

    def forward(self, x):
        # x: B 2C G mlp-> B C G  permute-> B G C
        if self.permute:
            return self.share_mlp(x).permute(0, 2, 1)
        else:
            return self.share_mlp(x)


## MLP
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# K_Norm + K_Pool + Shared MLP
class LNPBlock(nn.Module):
    def __init__(self, lga_out_dim, k_group_size, alpha, beta, mlp_in_dim, mlp_out_dim, num_group=128,
                 act_layer=nn.SiLU, drop_path=0., norm_layer=nn.LayerNorm, ):
        super().__init__()
        '''
        lga_out_dim: 2C
        mlp_in_dim: 2C
        mlp_out_dim: C
        x --->  (lga -> pool -> mlp -> act) --> x
        '''
        self.num_group = num_group
        self.lga_out_dim = lga_out_dim

        self.lga = K_Norm(self.lga_out_dim, k_group_size, alpha, beta)
        self.kpool = K_Pool()
        self.mlp = Post_ShareMLP(mlp_in_dim, mlp_out_dim)
        self.pre_norm_ft = norm_layer(self.lga_out_dim)

        self.act = act_layer()

    def forward(self, center, feat):
        # feat: B G+1 C
        B, G, C = feat.shape
        cls_token = feat[:, 0, :].view(B, 1, C)
        feat = feat[:, 1:, :]  # B G C

        lc_x_w = self.lga(center, feat)  # B 2C G K

        lc_x_w = self.kpool(lc_x_w)  # B 2C G : 1 768 128

        # norm([2C])
        lc_x_w = self.pre_norm_ft(lc_x_w.permute(0, 2, 1))  # pre-norm B G 2C
        lc_x = self.mlp(lc_x_w.permute(0, 2, 1))  # B G C : 1 128 384

        lc_x = self.act(lc_x)

        lc_x = torch.cat((cls_token, lc_x), dim=1)  # B G+1 C : 1 129 384
        return lc_x


class MambaBlock(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.SiLU,
                 norm_layer=nn.LayerNorm,
                 k_group_size=8,
                 alpha=100,
                 beta=1000,
                 num_group=128,
                 num_heads=6,
                 bimamba_type="v2",
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.num_group = num_group
        self.k_group_size = k_group_size

        self.num_heads = num_heads

        self.lfa = LNPBlock(lga_out_dim=dim * 2,
                            k_group_size=self.k_group_size,
                            alpha=alpha,
                            beta=beta,
                            mlp_in_dim=dim * 2,
                            mlp_out_dim=dim,
                            num_group=self.num_group,
                            act_layer=act_layer,
                            drop_path=drop_path,
                            # num_heads=self.num_heads, # uncomment this line if use attention
                            norm_layer=norm_layer,
                            )

        self.mixer = Mamba(dim, bimamba_type=bimamba_type)

    def shuffle_x(self, x, shuffle_idx):
        pos = x[:, None, 0, :]
        feat = x[:, 1:, :]
        shuffle_feat = feat[:, shuffle_idx, :]
        x = torch.cat([pos, shuffle_feat], dim=1)
        return x

    def mamba_shuffle(self, x):
        G = x.shape[1] - 1  #
        shuffle_idx = torch.randperm(G)
        # shuffle_idx = torch.randperm(int(0.4*self.num_group+1)) # 1-mask
        x = self.shuffle_x(x, shuffle_idx)  # shuffle

        x = self.mixer(self.norm2(x))  # layernorm->mamba

        x = self.shuffle_x(x, shuffle_idx)  # un-shuffle
        return x

    def forward(self, center, x):
        # x + norm(x)->lfa(x)->dropout
        x = x + self.drop_path(self.lfa(center, self.norm1(x)))  # x: 32 129 384. center: 32 128 3

        # x + norm(x)->mamba(x)->dropout
        # x = x + self.drop_path(self.mamba_shuffle(x))
        x = x + self.drop_path(self.mixer(self.norm2(x)))

        return x


class MambaEncoder(nn.Module):
    def __init__(self, k_group_size=8, embed_dim=768, depth=4, drop_path_rate=0., num_group=128, num_heads=6,
                 bimamba_type="v2", ):
        super().__init__()
        self.num_group = num_group
        self.k_group_size = k_group_size
        self.num_heads = num_heads
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=embed_dim,  #
                k_group_size=self.k_group_size,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,  #
                num_group=self.num_group,
                num_heads=self.num_heads,
                bimamba_type=bimamba_type,
            )
            for i in range(depth)])

    def forward(self, center, x, pos):
        '''
        INPUT:
            x: patched point cloud and encoded, B G+1 C, 8 128+1=129 384
            pos: positional encoding, B G+1 C, 8 128+1=129 384
        OUTPUT:
            x: x after transformer block, keep dim, B G+1 C, 8 128+1=129 384

        NOTE: Remember adding positional encoding for every block, 'cause ptc is sensitive to position
        '''
        # TODO: pre-compute knn (GroupFeature)
        for _, block in enumerate(self.blocks):
            x = block(center, x + pos)
        return x


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        # points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = self.square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class MambaMesh(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.trans_dim = config['trans_dim']
        self.depth = config['depth']
        self.drop_path_rate = config['drop_path_rate']
        self.cls_dim = config['cls_dim']
        self.num_heads = config['num_heads']

        self.group_size = config['group_size']   
        self.num_group = config['num_group']
        self.encoder_dims = config['encoder_dims']
        self.input_channel = config['input_channel']

        self.FaceTex_Extractor = FaceTex_Extractor()
        self.Spatial_Extractor = Spatial_Extractor()
        self.FaceShape_Extractor = FaceShape_Extractor()
        self.Structural_Extractor = Structural_Extractor(num_kernel=64, num_neighbor=3)

        self.encoder = Encoder(input_channel=self.input_channel, encoder_channel=self.encoder_dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.SiLU(),
            nn.Linear(128, self.trans_dim)
        )

        self.ordering = config['ordering']
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.k_group_size = config['center_local_k']  # default=8

        self.bimamba_type = config['bimamba_type']

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        # define the encoder
        self.blocks = MambaEncoder(
            embed_dim=self.trans_dim,
            k_group_size=self.k_group_size,
            depth=self.depth,
            drop_path_rate=dpr,
            num_group=self.num_group,
            num_heads=self.num_heads,
            bimamba_type=self.bimamba_type,
        )
        # embed_dim=768, depth=4, drop_path_rate=0.

        self.norm = nn.LayerNorm(self.trans_dim)

        self.FPUP = FeaturePropagation(self.trans_dim, [64, 64, 64])

        # saliency map head
        # self.fineture = nn.Sequential(
        #     nn.Conv1d(64, 32, 1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 32, 1),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.Conv1d(32, 1, 1),
        #     nn.Sigmoid(),
        # )

        # classification head
        self.fineture = nn.Sequential(
            nn.Conv1d(self.trans_dim, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, config['cls_dim'])
        )

        self.label_smooth = config['label_smooth']
        self.build_loss_func()

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss(label_smoothing=self.label_smooth)

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', 'Transformer')
        else:
            print('Training from scratch!!!', 'Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, verts, faces, centers, normals, corners, neighbor_index,
                ring_1, ring_2, ring_3, face_colors, face_textures, texture, uv_grid):
        # Face Texture features
        tex_fea = self.FaceTex_Extractor(face_textures, texture, uv_grid)
        # Face Spatial features
        spatial_fea0 = self.Spatial_Extractor(centers)
        # Face Shape features
        shape_fea0 = self.FaceShape_Extractor(centers, ring_1, corners, verts, faces)
        # Face Curve features
        curve_fea0 = self.Structural_Extractor(normals, ring_1)

        feat_inp = torch.concatenate([tex_fea, spatial_fea0, shape_fea0, curve_fea0], dim=1)
        feat_inp = feat_inp.permute(0, 2, 1).contiguous()
        centers = centers.permute(0, 2, 1).contiguous()

        neighborhood, center, neighborhood_idx = self.group_divider(centers, ring_1, verts, faces)  # B G K 3
        feat_inp = index_points(feat_inp, neighborhood_idx)
        group_input_tokens = self.encoder(feat_inp)  # B G C

        cls_tokens = self.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = self.cls_pos.expand(group_input_tokens.size(0), -1, -1)

        pos = self.pos_embed(center)  # B G C

        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        # transformer
        x = self.blocks(center, x, pos)  # enter transformer blocks
        x = self.norm(x)

        x = x[:, 0].unsqueeze(1).expand(-1, self.num_group, -1) + x[:, 1:]
        concat_f = self.FPUP(centers, center, None, x)

        ret = self.fineture(concat_f)
        ret = ret.squeeze()
        return ret




        