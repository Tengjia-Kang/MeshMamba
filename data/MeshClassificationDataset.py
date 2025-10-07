import numpy as np
import os
import torch
import torch.utils.data as data
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh
from trimesh.graph import face_adjacency
from pytorch3d.transforms import Rotate, RotateAxisAngle, Transform3d
import math
from torchvision import transforms
import torch.nn.functional as F


class MeshClassificationDataset(data.Dataset):
    def __init__(self, cfg, part='train'):
        self.augment_data = cfg['augment_data'] if 'augment_data' in cfg else False
        if self.augment_data and part == "train":
            self.augment_vert = cfg.get('augment_vert', False)
            self.augment_rotation = cfg.get('augment_rotation', True)
        else:
            self.augment_vert = False
            self.augment_rotation = False

        self.device = torch.device('cpu:0')
        self.root = cfg['data_root']
        self.max_faces = cfg['max_faces']
        self.part = part
        
        if self.augment_data:
            self.jitter_sigma = cfg.get('jitter_sigma', 0.01)
            self.jitter_clip = cfg.get('jitter_clip', 0.05)

        # 获取所有类别名称
        self.categories = sorted(os.listdir(self.root))
        self.categories = [cat for cat in self.categories if os.path.isdir(os.path.join(self.root, cat))]
        
        # 创建类别到索引的映射
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # 收集所有数据文件路径和对应的标签
        self.data = []
        for category in self.categories:
            category_dir = os.path.join(self.root, category)
            if os.path.isdir(category_dir):
                # 遍历类别文件夹中的所有npz文件
                for file in os.listdir(category_dir):
                    if file.endswith('.npz'):
                        file_path = os.path.join(category_dir, file)
                        # 标签是类别的索引
                        label = self.category_to_idx[category]
                        self.data.append((file_path, label, category, file[:-4]))  # 去掉.npz后缀作为名称

        print(f"Loaded {len(self.data)} samples from {len(self.categories)} categories: {self.categories}")

    def __getitem__(self, i):
        file_path, label, category, mesh_name = self.data[i]
        
        # 加载npz文件
        data = np.load(file_path, allow_pickle=True)
        
        # 提取网格数据
        verts = data['verts'] if 'verts' in data else data['vertices']
        faces = data['faces']
        
        # 如果存在预计算的特征，则加载它们
        centers = data['centers'] if 'centers' in data else None
        normals = data['normals'] if 'normals' in data else None
        corners = data['corners'] if 'corners' in data else None
        
        # 如果没有预计算的特征，则计算它们
        if centers is None or normals is None or corners is None:
            # 计算面中心点
            if corners is None:
                corners = verts[faces]
            if centers is None:
                centers = np.mean(corners, axis=1)
            if normals is None:
                # 简单计算法向量（实际应用中可能需要更精确的方法）
                v1 = corners[:, 1, :] - corners[:, 0, :]
                v2 = corners[:, 2, :] - corners[:, 0, :]
                normals = np.cross(v1, v2)
                norm_magnitude = np.linalg.norm(normals, axis=1, keepdims=True)
                # 避免除以零
                norm_magnitude[norm_magnitude == 0] = 1
                normals = normals / norm_magnitude
        
        # 转换为tensor
        verts = torch.from_numpy(verts).float()
        faces = torch.from_numpy(faces).long()
        centers = torch.from_numpy(centers).float()
        normals = torch.from_numpy(normals).float()
        corners = torch.from_numpy(corners).float()
        
        # 数据增强
        if self.augment_data and self.part == 'train':
            # 顶点抖动
            if self.augment_vert:
                jittered_data = torch.clamp(
                    self.jitter_sigma * torch.randn_like(verts),
                    -self.jitter_clip, self.jitter_clip
                )
                verts = verts + jittered_data
            
            # 随机旋转
            if self.augment_rotation:
                rotation_matrix = self.get_random_rotation_matrix()
                R = Rotate(rotation_matrix, device=self.device)
                verts = R.transform_points(verts.unsqueeze(0)).squeeze(0)
                
                # 更新法向量
                normals = torch.matmul(normals, rotation_matrix.t())

        # 获取角向量
        if corners.dim() == 3 and corners.shape[1] == 3:
            corners = corners - centers.unsqueeze(1).repeat(1, 3, 1)
            corners = corners.view(corners.shape[0], -1)  # 展平为 (F, 9)

        # 创建标签tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # 字典用于批处理
        collated_dict = {
            'faces': faces,
            'verts': verts,
            'centers': centers,
            'normals': normals,
            'corners': corners,
            'label': label_tensor,
            'category': category,
            'mesh_name': mesh_name
        }
        
        return collated_dict

    def __len__(self):
        return len(self.data)

    def get_random_rotation_matrix(self):
        # 随机生成欧拉角
        theta_x = torch.rand(1).item() * 2 * math.pi  # 0 到 2π 的随机角度
        theta_y = torch.rand(1).item() * 2 * math.pi  # 0 到 2π 的随机角度
        theta_z = torch.rand(1).item() * 2 * math.pi  # 0 到 2π 的随机角度

        # 生成绕X轴的旋转矩阵
        R_x = torch.tensor([
            [1, 0, 0],
            [0, math.cos(theta_x), -math.sin(theta_x)],
            [0, math.sin(theta_x), math.cos(theta_x)]
        ], dtype=torch.float32)

        # 生成绕Y轴的旋转矩阵
        R_y = torch.tensor([
            [math.cos(theta_y), 0, math.sin(theta_y)],
            [0, 1, 0],
            [-math.sin(theta_y), 0, math.cos(theta_y)]
        ], dtype=torch.float32)

        # 生成绕Z轴的旋转矩阵
        R_z = torch.tensor([
            [math.cos(theta_z), -math.sin(theta_z), 0],
            [math.sin(theta_z), math.cos(theta_z), 0],
            [0, 0, 1]
        ], dtype=torch.float32)

        # 合成最终的旋转矩阵
        R = torch.matmul(torch.matmul(R_z, R_y), R_x)
        return R

    def get_categories(self):
        """返回类别列表"""
        return self.categories
    
    def get_num_classes(self):
        """返回类别数量"""
        return len(self.categories)