"""
Data is pre-processed to obtain the following infomation:
    1) Vertices and Faces of the mesh
    2) 1 Ring, 2 Ring, and 3 Ring neighborhood of the mesh faces
"""
import os
import argparse
import numpy as np
import torch
from pytorch3d.structures import Meshes
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh
from trimesh.graph import face_adjacency
from utils.file_utils import fpath
from utils.mesh_utils import pytorch3D_mesh, is_mesh_valid, normalize_mesh

# 设置命令行参数
parser = argparse.ArgumentParser(description='预处理OBJ文件，生成包含网格信息的NPZ文件')
parser.add_argument('--input_dir', type=str, default='dataset/Manifold40/', help='输入OBJ文件所在的根目录')
parser.add_argument('--output_dir', type=str, default=None, help='输出NPZ文件的根目录')
parser.add_argument('--device', type=str, default='cpu:0', help='计算设备 (cpu 或 cuda)')
parser.add_argument('--max_faces', type=int, default=500, help='最大面数')
args = parser.parse_args()

device = torch.device(args.device)
data_root = args.input_dir
output_root = args.output_dir
max_faces = args.max_faces

# 如果未指定输出目录，则使用输入目录
if output_root is None:
    output_root = data_root

if not os.path.exists(data_root):
    raise Exception('数据集未在 {0} 找到'.format(data_root))

# 确保输出目录存在
if not os.path.exists(output_root):
    os.makedirs(output_root)

fpath_data = fpath(data_root)

for path in fpath_data:
    try:
        mesh, faces, verts, edges, v_normals, f_normals = pytorch3D_mesh(path, device)
        if not is_mesh_valid(mesh):
            print(f'警告: 网格 {path} 无效，跳过处理。')
            continue
        
        # 检查面数是否符合预期
        if faces.shape[0] != max_faces:
            print(f'警告: {path} 的面数 {faces.shape[0]} 不符合预期的 {max_faces}，跳过处理。')
            continue

        # 归一化网格
        mesh, faces, verts, edges, v_normals, f_normals = normalize_mesh(verts=verts, faces=faces)

        ########################################################################### 1st-Ring ###########################################################################
        data = Data(pos=verts, edge_index=edges.permute(1, 0), face=faces.permute(1, 0))
        trimesh = to_trimesh(data)
        # 沿边的相邻面索引，沿相邻面的边
        faces_adjacency, edges_adjacency = face_adjacency(faces=faces.permute(1, 0),
                                                          mesh=trimesh,
                                                          return_edges=True)

        faces_neighbor_1st_ring = []
        edges_neighbor_1ring = []

        # 对每个面获取其沿边的1-Ring邻域
        # 对每个面获取面与相邻面之间的边
        for face_idx in range(max_faces):
            face_dim_0 = np.argwhere(faces_adjacency[:, 0] == face_idx)
            face_dim_1 = np.argwhere(faces_adjacency[:, 1] == face_idx)

            face_neighbor_dim_0 = faces_adjacency[:, 0][face_dim_1]
            face_neighbor_dim_1 = faces_adjacency[:, 1][face_dim_0]

            face_neighbor_1st_ring = np.concatenate([face_neighbor_dim_0,
                                                     face_neighbor_dim_1])

            # 面与相邻面之间的边
            face_edge = np.concatenate([face_dim_0, face_dim_1]).reshape(-1)
            edge_neighbor_1ring = edges_adjacency[face_edge]

            faces_neighbor_1st_ring.insert(face_idx, face_neighbor_1st_ring)
            edges_neighbor_1ring.insert(face_idx, edge_neighbor_1ring)

        faces_neighbor_1st_ring = np.asarray(faces_neighbor_1st_ring).squeeze(2)
        edges_neighbor_1ring = np.asarray(edges_neighbor_1ring)

        # 每个面在1st Ring中连接到3个其他面
        assert faces_neighbor_1st_ring.shape == (max_faces, 3)
        # 每个面与相邻面之间有1条边
        # 最后一维为2，因为每条边由2个顶点组成
        assert edges_neighbor_1ring.shape == (max_faces, 3, 2)

        ########################################################################### 2nd-Ring ###########################################################################
        faces_neighbor_0th_ring = np.arange(max_faces)
        faces_neighbor_2ring = faces_neighbor_1st_ring[faces_neighbor_1st_ring]
        faces_neighbor_0ring = np.stack([faces_neighbor_0th_ring]*3, axis=1)
        faces_neighbor_0ring = np.stack([faces_neighbor_0ring]*3, axis=2)

        dilation_mask = faces_neighbor_2ring != faces_neighbor_0ring
        faces_neighbor_2nd_ring = faces_neighbor_2ring[dilation_mask]
        faces_neighbor_2nd_ring = faces_neighbor_2nd_ring.reshape(max_faces, -1)

        # 每个面在其2-Ring邻域中有6个相邻面
        assert faces_neighbor_2nd_ring.shape == (max_faces, 6)

        ########################################################################### 3rd-Ring ###########################################################################
        faces_neighbor_3ring = faces_neighbor_2nd_ring[faces_neighbor_1st_ring]
        faces_neighbor_3ring = faces_neighbor_3ring.reshape(max_faces, -1)

        faces_neighbor_3rd_ring = []
        for face_idx in range(max_faces):
            face_neighbor_3ring = faces_neighbor_3ring[face_idx]
            for neighbor in range(3):
                face_neighbor_1st_ring = faces_neighbor_1st_ring[face_idx, neighbor]
                dilation_mask = np.delete(
                    np.arange(face_neighbor_3ring.shape[0]),
                    np.where(face_neighbor_3ring == face_neighbor_1st_ring)[0][0:2])
                face_neighbor_3ring = face_neighbor_3ring[dilation_mask]
            faces_neighbor_3rd_ring.insert(face_idx, face_neighbor_3ring)
        # 每个面在其3-Ring邻域中有12个相邻面
        faces_neighbor_3rd_ring = np.array(faces_neighbor_3rd_ring)
        assert faces_neighbor_3rd_ring.shape == (max_faces, 12)

        corners = verts[faces.long()]
        # 每个面在1st Ring中连接到3个其他面
        assert corners.shape == (max_faces, 3, 3)

        centers = torch.sum(corners, axis=1)/3
        assert centers.shape == (max_faces, 3)

        corners = corners.reshape(-1, 9)
        assert f_normals.shape == (max_faces, 3)

        faces_feature = np.concatenate([centers, corners, f_normals], axis=1)
        assert faces_feature.shape == (max_faces, 15)

        # 计算相对于数据根目录的相对路径
        rel_path = os.path.relpath(path, data_root)
        # 构建输出文件路径
        output_path = os.path.join(output_root, rel_path.replace('.obj', '.npz'))
        # 创建输出目录
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存NPZ文件
        np.savez(output_path,
                 verts=verts,
                 faces=faces,
                 ring_1=faces_neighbor_1st_ring,
                 ring_2=faces_neighbor_2nd_ring,
                 ring_3=faces_neighbor_3rd_ring)
        
        print(f'已成功处理: {path}')
        print(f'输出文件: {output_path}')
        
    except Exception as e:
        print(f'处理 {path} 时出错: {e}')
        continue

print('所有文件处理完成！')
