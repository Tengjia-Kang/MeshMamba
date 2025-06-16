import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import torchvision.models as models


class FaceTex_Extractor(nn.Module):
    def __init__(self):
        super(FaceTex_Extractor, self).__init__()
        # backbone = models.resnet34(models.ResNet34_Weights)
        # self.extractor = nn.Sequential(backbone.conv1,
        #                                backbone.bn1,
        #                                backbone.relu,
        #                                backbone.layer1)
        self.texture_conv = nn.Sequential(
            nn.Conv2d(3, 16, 7, stride=1, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.tex_conv = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=3, padding=1),
            nn.Conv2d(32, 64, 1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)
        )

    def forward(self, face_textures, texture, uv_grid):
        texture = self.texture_conv(texture)
        # texture = self.extractor(texture)
        # grid the sample
        texture = texture.unsqueeze(2)
        face_textures = F.grid_sample(texture, grid=uv_grid, mode='bilinear', align_corners=True)
        face_textures = face_textures.transpose(1, 2)

        B, N, C, H, W = face_textures.shape
        face_textures = face_textures.reshape(B*N, C, H, W)
        fea = self.tex_conv(face_textures)
        fea = fea.squeeze().reshape(B, N, -1)
        fea = fea.permute(0, 2, 1)

        return fea


class Spatial_Extractor(nn.Module):
    def __init__(self):
        super(Spatial_Extractor, self).__init__()

        self.spatial_mlp = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

    def forward(self, centers):
        return self.spatial_mlp(centers)


class FaceShape_Extractor(nn.Module):
    def __init__(self):
        super(FaceShape_Extractor, self).__init__()
        self.num_neighbor = 3
        self.rotate_mlp = nn.Sequential(
            nn.Conv1d(6, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.fusion_mlp = nn.Sequential(
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

    def forward(self, centers, ring_n, corners, verts, faces):
        # take the centers of neighbor faces by index ring_n
        centers = centers.permute(0, 2, 1)
        centers_exp = centers.unsqueeze(2).expand(-1, -1, self.num_neighbor, -1)
        ring_n_exp = ring_n.unsqueeze(3).expand(-1, -1, -1, 3)
        centers_ring = torch.gather(centers_exp, 1, ring_n_exp)
        center_vectors = centers_ring - centers.unsqueeze(3)
        center_vectors = center_vectors.flatten(2, 3)
        center_vectors = center_vectors.permute(0, 2, 1)
        # normals_ring = torch.cat([normals_ring, centers.unsqueeze(2)], 2)

        # vertices gather
        verts_exp = verts.unsqueeze(2).expand(-1, -1, 3, -1)
        faces_exp = faces.unsqueeze(3).expand(-1, -1, -1, 3)
        verts_face = torch.gather(verts_exp, 1, faces_exp)
        verts_face = verts_face.flatten(2, 3)
        verts_face = verts_face.permute(0, 2, 1)

        fea = (self.rotate_mlp(corners[:, :6]) +
               self.rotate_mlp(corners[:, 3:9]) +
               self.rotate_mlp(torch.cat([corners[:, 6:], corners[:, :3]], 1))) / 3
        return self.fusion_mlp(fea)


class Structural_Extractor(nn.Module):
    def __init__(self, num_kernel, num_neighbor=3):
        super(Structural_Extractor, self).__init__()
        self.num_kernel = num_kernel
        self.num_neighbor = num_neighbor
        self.conv = nn.Conv1d(3, 64, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(self.num_kernel)

    def forward(self, normals, ring_n):
        # take the normals of neighbor faces by index ring_n
        normals = normals.permute(0, 2, 1)
        normals_exp = normals.unsqueeze(2).expand(-1, -1, self.num_neighbor, -1)
        ring_n_exp = ring_n.unsqueeze(3).expand(-1, -1, -1, 3)
        normals_ring = torch.gather(normals_exp, 1, ring_n_exp)
        # normals_ring = torch.cat([normals_ring, normals.unsqueeze(2)], 2)
        neighbor_direction_norm = F.normalize(normals_ring, dim=-1)
        center_direction_norm = F.normalize(normals_exp, dim=-1)
        # support_direction_norm = F.normalize(self.directions, dim=0)
        # feature = neighbor_direction_norm @ support_direction_norm

        feature = torch.mul(neighbor_direction_norm, center_direction_norm)

        # feature = neighbor_direction_norm @ center_direction_norm
        # assert feature.shape == (num_meshes, num_faces, self.num_samples, self.num_kernel)

        feature = torch.min(feature, dim=2)[0]
        feature = feature.permute(0, 2, 1)
        feature = self.relu(self.bn(self.conv(feature)))
        return feature


class GraphConvolution(nn.Module):
    def __init__(self, num_kernel, num_neighbor=3):
        super(GraphConvolution, self).__init__()
        self.num_neighbor = num_neighbor
        self.fea_in_channel = num_kernel
        self.num_kernel = num_kernel
        self.num_matrix = 16
        self.Weight_Matrix = nn.Parameter(torch.FloatTensor(self.fea_in_channel, self.num_kernel, self.num_matrix))
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm1d(self.num_kernel)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.num_kernel)
        self.Weight_Matrix.data.uniform_(-stdv, stdv)

    def forward(self, fea, ring_n):
        # take the normals of neighbor faces by index ring_n
        fea = fea.permute(0, 2, 1)
        fea_exp = fea.unsqueeze(2).expand(-1, -1, self.num_neighbor, -1)
        ring_n_exp = ring_n.unsqueeze(3).expand(-1, -1, -1, self.fea_in_channel)
        fea_ring = torch.gather(fea_exp, 1, ring_n_exp)
        fea_ring = torch.cat([fea_ring, fea.unsqueeze(2)], 2)
        # neighbor_direction_norm = F.normalize(fea_ring, dim=-1)
        # support_direction_norm = F.normalize(self.directions, dim=0)
        # feature = fea_ring @ self.Weight_Matrix
        # feature = torch.matmul(fea_ring, self.Weight_Matrix)
        # assert feature.shape == (num_meshes, num_faces, self.num_samples, self.num_kernel)
        res = torch.matmul(fea_ring, self.Weight_Matrix.view(self.fea_in_channel, -1))  # 结果将是 [B, F, N, 1, E * M]
        result = res.view(res.shape[0], res.shape[1], res.shape[2], self.num_kernel, self.num_matrix)  # 再reshape成 [B, F, N, E, M]

        feature = torch.mean(result, dim=2)
        feature = torch.mean(feature, dim=3)
        feature = feature.permute(0, 2, 1)
        feature = self.relu(self.bn(feature))
        return feature
