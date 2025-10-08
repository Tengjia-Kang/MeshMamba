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
from utils.Manifold40 import model_net_labels

class MeshClassificationDataset(data.Dataset):
    def __init__(self, cfg, part='train'):
        self.augment_data = cfg['augment_data'] if 'augment_data' in cfg else False
        if self.augment_data and part == "train":
            self.augment_vert = cfg.get('augment_vert', False)
            self.augment_rotation = cfg.get('augment_rotation', True)
        else:
            self.augment_vert = False
            self.augment_rotation = False

        self.device = torch.device(f"cuda:{cfg['devices']}")
        self.root = cfg['data_root']
        self.max_faces = cfg['max_faces']
        self.part = part
        
        if self.augment_data:
            self.jitter_sigma = cfg.get('jitter_sigma', 0.01)
            self.jitter_clip = cfg.get('jitter_clip', 0.05)

        # 获取所有类别名称
        self.categories = sorted(model_net_labels)

        self.num_classes = len(self.categories)
        # 创建类别到索引的映射
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        
        # 收集所有数据文件路径和对应的标签
        self.data = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(1024),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=10)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        for category in self.categories:
            category_dir = os.path.join(self.root, category)
            if os.path.isdir(category_dir):
                # 遍历类别文件夹中的所有npz文件
                for file in os.listdir(os.path.join(category_dir, part)):
                    if file.endswith('.npz'):
                        file_path = os.path.join(category_dir,part, file)
                        # 标签是类别的索引
                        label = self.category_to_idx[category]
                        self.data.append((file_path, label, category, file[:-4]))  # 去掉.npz后缀作为名称

        print(f"Loaded {len(self.data)} samples from {len(self.categories)} categories: {self.categories}")

    def __getitem__(self, i):
        file_path, label, category, mesh_name = self.data[i]
        
        # 加载npz文件
        data = np.load(file_path, allow_pickle=True)
        


        # 提取网格数据
        verts = data['verts']
        centers = data['centers']
        normals = data['f_normals']
        corners = data['corners']
        
        # 如果存在预计算的特征，则加载它们
        faces = data['faces']
        ring_1 = data['ring_1']
        ring_2 = data['ring_2']
        ring_3 = data['ring_3']
        neighbors = data['neighbors']
        label = data['label']


        # 如果没有预计算的特征，则计算它们
        # if centers is None or normals is None or corners is None:
        #     # 计算面中心点
        #     if corners is None:
        #         corners = verts[faces]
        #     if centers is None:
        #         centers = np.mean(corners, axis=1)
        #     if normals is None:
        #         # 简单计算法向量（实际应用中可能需要更精确的方法）
        #         v1 = corners[:, 1, :] - corners[:, 0, :]
        #         v2 = corners[:, 2, :] - corners[:, 0, :]
        #         normals = np.cross(v1, v2)
        #         norm_magnitude = np.linalg.norm(normals, axis=1, keepdims=True)
        #         # 避免除以零
        #         norm_magnitude[norm_magnitude == 0] = 1
        #         normals = normals / norm_magnitude
        
        # current_face_num = faces.shape[0]
        # # 1. 处理 faces (形状: [N, 3] → [max_faces, 3])
        # if current_face_num > self.max_faces:
        #     # 超过最大面数：截断
        #     faces = faces[:self.max_faces]
        # else:
        #     # 不足最大面数：补零（用0填充）
        #     pad_faces = np.zeros((self.max_faces - current_face_num, 3), dtype=faces.dtype)
        #     faces = np.concatenate([faces, pad_faces], axis=0)

        # if centers is not None:
        #     if len(centers.shape) == 2 and centers.shape[0] == current_face_num:
        #         # 形状 [N, 3] → 截断/补零到 [max_faces, 3]
        #         if current_face_num > self.max_faces:
        #             centers = centers[:self.max_faces]
        #         else:
        #             pad_centers = np.zeros((self.max_faces - current_face_num, 3), dtype=centers.dtype)
        #             centers = np.concatenate([centers, pad_centers], axis=0)
        #     elif len(centers.shape) == 2 and centers.shape[1] == current_face_num:
        #         # 若原始是 [3, N]，先转置为 [N, 3] 再处理
        #         centers = centers.T  # [N, 3]
        #         if current_face_num > self.max_faces:
        #             centers = centers[:self.max_faces]
        #         else:
        #             pad_centers = np.zeros((self.max_faces - current_face_num, 3), dtype=centers.dtype)
        #             centers = np.concatenate([centers, pad_centers], axis=0)
        #         centers = centers.T  # 转回 [3, max_faces]（如果模型需要此格式）
        
        # # 3. 处理 normals (假设形状: [N, 3] → [max_faces, 3])
        # if len(normals.shape) == 2 and normals.shape[0] == current_face_num:
        #     if current_face_num > self.max_faces:
        #         normals = normals[:self.max_faces]
        #     else:
        #         pad_normals = np.zeros((self.max_faces - current_face_num, 3), dtype=normals.dtype)
        #         normals = np.concatenate([normals, pad_normals], axis=0)
        
        # # 4. 处理 corners (假设形状: [N, 9] 或 [N, 3, 3] → 统一为 [max_faces, ...])
        # if corners is not None:
        #     if corners.shape[0] == current_face_num:
        #         if current_face_num > self.max_faces:
        #             corners = corners[:self.max_faces]
        #         else:
        #             # 补零（根据原始形状补全，这里假设是 [N, 9]）
        #             pad_shape = list(corners.shape[1:])
        #             pad_corners = np.zeros([self.max_faces - current_face_num] + pad_shape, dtype=corners.dtype)
        #             corners = np.concatenate([corners, pad_corners], axis=0)
        
        # # 5. 处理 ring_1/ring_2/ring_3 (形状: [N] → [max_faces])
        # # for ring in [ring_1, ring_2, ring_3]:
        # #     if len(ring) == current_face_num:
        # #         if current_face_num > self.max_faces:
        # #             ring = ring[:self.max_faces]
        # #         else:
        # #             # 补-1（表示无效索引，避免与有效面索引冲突）
        # #             pad_ring = np.full((self.max_faces - current_face_num,), -1, dtype=ring.dtype)
        # #             ring = np.concatenate([ring, pad_ring], axis=0)
        
        # # 6. 处理 neighbors (假设形状: [N, K] → [max_faces, K])

        # if neighbors.ndim == 2 and neighbors.shape[0] == current_face_num:
        #     K = neighbors.shape[1]  # 每个面的邻域数
        #     if current_face_num > self.max_faces:
        #         neighbors = neighbors[:self.max_faces]
        #     else:
        #         # 补-1（无效索引）
        #         pad_neighbors = np.full((self.max_faces - current_face_num, K), -1, dtype=neighbors.dtype)
        #         neighbors = np.concatenate([neighbors, pad_neighbors], axis=0)
        # ---------------------------------------------------------------------------------


        # Convert to tensor
        faces = torch.from_numpy(faces).long()
        ring_1 = torch.from_numpy(ring_1).long()
        ring_2 = torch.from_numpy(ring_2).long()
        ring_3 = torch.from_numpy(ring_3).long()
        neighbors = torch.from_numpy(neighbors).long()
        # verts = verts.float()
        # centers = centers.float()
        # normals = normals.float()

        # # 获取角向量
        # if corners.dim() == 3 and corners.shape[1] == 3:
        #     corners = corners - centers.unsqueeze(1).repeat(1, 3, 1)
        #     corners = corners.view(corners.shape[0], -1)  # 展平为 (F, 9)

        # corners = corners.float()
        
 
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


        # 创建标签tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        verts = torch.from_numpy(verts).float()
        centers = torch.from_numpy(centers).float()
        normals = torch.from_numpy(normals).float()
        corners = torch.from_numpy(corners).float()
        # 字典用于批处理
        collated_dict = {
            'faces': faces,
            'verts': verts,
            'centers': centers,
            'normals': normals,
            'corners': corners,
            'neighbors': neighbors,
            'ring_1': ring_1,
            'ring_2': ring_2,
            'ring_3': ring_3,
            'label': label_tensor,
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
    



    def collect_data(self, path):
        mesh, faces, verts, edges, v_normals, f_normals = self.pytorch3D_mesh(path, self.device)
        max_faces = faces.shape[0]
        if not self.is_mesh_valid(mesh):
            raise ValueError('Mesh is invalid!')
        assert faces.shape[0] == (max_faces)

        # Normalize Mesh
        # mesh, faces, verts, edges, v_normals, f_normals = normalize_mesh(verts=verts, faces=faces)

        # move to center
        center = (torch.max(verts, 0)[0] + torch.min(verts, 0)[0]) / 2
        verts -= center

        # normalize
        max_len = torch.max(verts[:, 0] ** 2 + verts[:, 1] ** 2 + verts[:, 2] ** 2)
        verts /= torch.sqrt(max_len)

        # get corners
        corners = verts[faces.long()]
        # Each face is connected to 3 other faces in the 1st Ring
        assert corners.shape == (max_faces, 3, 3)

        centers = torch.sum(corners, axis=1) / 3
        assert centers.shape == (max_faces, 3)

        corners = corners.reshape(-1, 9)
        assert f_normals.shape == (max_faces, 3)

        faces_feature = np.concatenate([centers, corners, f_normals], axis=1)
        assert faces_feature.shape == (max_faces, 15)

        max_ver = 500
        verts = np.pad(verts, ((0, max_ver - verts.shape[0]), (0, 0)), mode='constant')
        verts = torch.from_numpy(verts)

        collated_dict = {
            'faces': faces,
            'verts': verts,
            'centers': centers,
            'normals': f_normals,
            'corners': corners,
        }
        return collated_dict
    