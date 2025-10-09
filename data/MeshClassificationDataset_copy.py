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

model_net_labels = [
    'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
    'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
    'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
    'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
]
model_net_labels.sort()


class MeshClassificationDataset(data.Dataset):
    def __init__(self, cfg, part='train', mesh_paths=None):

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
        self.max_ver = cfg['max_ver']
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
        for mesh_path in mesh_paths:
            if mesh_path.endswith('.npz') or mesh_path.endswith('.obj'):
                mesh_name = mesh_path.split("/")[-1].split(".")[0]
                target = mesh_path.split('/')[-3]
                label = model_net_labels.index(target)
                npz_name = os.path.join("/home/ktj/Projects/MeshMamba/dataset/processed/Manifold_ringn", mesh_name+".npz")
                self.data.append((mesh_path, npz_name, mesh_name, label))

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(1024),
            # transforms.GaussianBlur(kernel_size=(3, 3), sigma=10)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        print(f"Loaded {len(self.data)} samples from {len(self.categories)} categories: {self.categories}")

    def __getitem__(self, i):
        mesh_path, npz_name, mesh_name, label = self.data[i]
        mesh = self.collect_data(mesh_path)
        ringn = np.load(npz_name)

        # 加载npz文件
        # data = np.load(file_path, allow_pickle=True)
        
        # position data
        verts = mesh['verts']
        centers = mesh['centers']
        normals = mesh['normals']
        corners = mesh['corners']
        
        # neighbor data
        faces = ringn['faces']
        ring_1 = ringn['ring_1']
        ring_2 = ringn['ring_2']
        ring_3 = ringn['ring_3']
        neighbors = ringn['neighbors']

        # data augmentation for training
        if self.augment_data and self.part == 'train':
            # vertex jittering
            if self.augment_vert:
                jittered_data = torch.clamp(
                    self.jitter_sigma * torch.randn_like(verts),
                    -self.jitter_clip, self.jitter_clip
                )
                verts = verts + jittered_data 
            # random rotation
            if self.augment_rotation:
                rotation_matrix = self.get_random_rotation_matrix()
                R = Rotate(rotation_matrix, device=self.device)
                verts = R.transform_points(verts.unsqueeze(0)).squeeze(0)
                normals = torch.matmul(normals, rotation_matrix.t())

        # Convert to tensor
        faces = torch.from_numpy(faces).long()
        ring_1 = torch.from_numpy(ring_1).long()
        ring_2 = torch.from_numpy(ring_2).long()
        ring_3 = torch.from_numpy(ring_3).long()
        neighbors = torch.from_numpy(neighbors).long()
        # 创建标签tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        # verts = torch.from_numpy(verts).float()
        # centers = torch.from_numpy(centers).float()
        # normals = torch.from_numpy(normals).float()
        # corners = torch.from_numpy(corners).float()
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
    
    def is_mesh_valid(self, mesh):
        """
        Check validity of pytorch3D mesh

        Args:
            mesh: pytorch3D mesh

        Returns:
            validity: validity of the mesh
        """
        validity = True

        # Check if the mesh is not empty
        if mesh.isempty():
            validity = False

        # Check if vertices in the mesh are valid
        verts = mesh.verts_packed()
        if not torch.isfinite(verts).all() or torch.isnan(verts).all():
            validity = False

        # Check if vertex normals in the mesh are valid
        v_normals = mesh.verts_normals_packed()
        if not torch.isfinite(v_normals).all() or torch.isnan(v_normals).all():
            validity = False

        # Check if face normals in the mesh are valid
        f_normals = mesh.faces_normals_packed()
        if not torch.isfinite(f_normals).all() or torch.isnan(f_normals).all():
            validity = False

        return validity

    def pytorch3D_mesh(self, f_path, device):
        """
        Read pytorch3D mesh from path

        Args:
            f_path: obj file path

        Returns:
            mesh, faces, verts, edges, v_normals, f_normals: pytorch3D mesh and other mesh information
        """
        if not f_path.endswith('.obj'):
            raise ValueError('Input files should be in obj format.')
        mesh = load_objs_as_meshes([f_path], device)
        # 应用随机旋转变换到网格
        angel = self.get_random_rotation_matrix()
        R = Rotate(angel, device=device)
        if self.augment_rotation and self.part == "train":
            mesh = mesh.update_padded(R.transform_points(mesh.verts_padded()))
        faces = mesh.faces_packed()
        verts = mesh.verts_packed()
        # edges = mesh.edges_packed()
        # v_normals = mesh.verts_normals_packed()
        f_normals = mesh.faces_normals_packed()
        # data augmentation
        if self.augment_vert and self.part == 'train':
            # jitter 中心点坐标加噪
            jittered_data = np.clip(self.jitter_sigma * np.random.randn(*verts.shape),
                                    -1 * self.jitter_clip, self.jitter_clip)  # clip截取区间值
            verts = verts + jittered_data
                # max_ver = 252
        
        # 填充顶点到最大顶点数 pad vertex to max_ver
        pad_amount = self.max_ver - verts.shape[0]
        if pad_amount > 0:
            # 使用 PyTorch 的 pad 函数，注意格式是 (0, 0, 0, pad_amount)
            # 表示在最后一个维度的前后各填充0，在倒数第二个维度的前面填充0、后面填充pad_amount
            verts = torch.nn.functional.pad(verts, (0, 0, 0, pad_amount), mode='constant', value=0)
        elif pad_amount < 0:
            # 如果顶点数量超过最大值，截断到指定长度
            verts = verts[:self.max_ver, :]
        # verts = torch.from_numpy(verts)
        # return mesh, faces, verts, edges, v_normals, f_normals
        return mesh, faces, verts, f_normals

    def collect_data(self, path):
        mesh, faces, verts, f_normals = self.pytorch3D_mesh(path, self.device)
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

        # faces_feature = np.concatenate([centers, corners, f_normals], axis=1)
        # assert faces_feature.shape == (max_faces, 15)

        collated_dict = {
            'faces': faces,
            'verts': verts,
            'centers': centers,
            'normals': f_normals,
            'corners': corners,
        }
        return collated_dict
    