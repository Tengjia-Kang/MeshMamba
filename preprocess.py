"""
Data is pre-processed to obtain the following infomation:
    1) Vertices and Faces of the mesh
    2) 1 Ring, 2 Ring, and 3 Ring neighborhood of the mesh faces
"""


import os
import numpy as np
import torch
from pytorch3d.structures import Meshes
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh
from trimesh.graph import face_adjacency
from utils.file_utils import fpath
from utils.mesh_utils import pytorch3D_mesh, is_mesh_valid, normalize_mesh
from pytorch3d.io import load_objs_as_meshes
import torchvision.transforms as transforms
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
# import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn.functional as F

def pytorch3D_mesh(f_path, device):
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
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    edges = mesh.edges_packed()
    v_normals = mesh.verts_normals_packed()
    f_normals = mesh.faces_normals_packed()

    return mesh, faces, verts, edges, v_normals, f_normals


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face

def get_manifold40_label(path):
    """
    Get Manifold40 label from path

    Args:
        path: obj file path

    Returns:
        label: Manifold40 label
    """
    model_net_labels = [
        'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet',
        'wardrobe', 'bookshelf', 'laptop', 'door', 'lamp', 'person', 'curtain', 'piano', 'airplane', 'cup',
        'cone', 'tent', 'radio', 'stool', 'range_hood', 'car', 'sink', 'guitar', 'tv_stand', 'stairs',
        'mantel', 'bench', 'plant', 'bottle', 'bowl', 'flower_pot', 'keyboard', 'vase', 'xbox', 'glass_box'
    ]
    model_net_labels.sort()
    target = path.split('/')[-3]
    label = model_net_labels.index(target)
     
    return label

# def fpath(dir_name):
#     """
#     Return all obj file in a directory

#     Args:
#         dir_name: root path to obj files

#     Returns:
#         f_path: list of obj files paths
#     """
#     f_path = []
#     for root, dirs, files in os.walk(dir_name, topdown=False):
#         for f in files:
#             if f.endswith('.obj'):
#                 if os.path.exists(os.path.join(root, f)):
#                     f_path.append(os.path.join(root, f))
#     return f_path


def normalize_mesh(verts, faces):
    """
    Normalize and center input mesh to fit in a sphere of radius 1 centered at (0,0,0)

    Args:
        mesh: pytorch3D mesh

    Returns:
        mesh, faces, verts, edges, v_normals, f_normals: normalized pytorch3D mesh and other mesh
        information
    """
    verts = verts - verts.mean(0)
    scale = max(verts.abs().max(0)[0])
    verts = verts / scale
    mesh = Meshes(verts=[verts], faces=[faces])
    faces = mesh.faces_packed().squeeze(0)
    verts = mesh.verts_packed().squeeze(0)
    edges = mesh.edges_packed().squeeze(0)
    v_normals = mesh.verts_normals_packed().squeeze(0)
    f_normals = mesh.faces_normals_packed().squeeze(0)

    return mesh, faces, verts, edges, v_normals, f_normals

def fpath(dir_name):
    """
    Return all obj file in a directory

    Args:
        dir_name: root path to obj files

    Returns:
        f_path: list of obj files paths
    """
    f_path = []
    for root, dirs, files in os.walk(dir_name, topdown=False):
        for f in files:
            if f.endswith('.obj'):
                if os.path.exists(os.path.join(root, f)):
                    f_path.append(os.path.join(root, f))
    return f_path



def main():
    device = torch.device('cpu')
    data_root = "/mnt/newdisk/ktj/Mesh/Manifold40/"
    output_root = '/mnt/newdisk/ktj/Mesh/dataset_preprocessed/Manifold40/'
    max_faces = 500

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

            # move to center
            center = (torch.max(verts, 0)[0] + torch.min(verts, 0)[0]) / 2
            verts -= center

            # normalize
            max_len = torch.max(verts[:, 0] ** 2 + verts[:, 1] ** 2 + verts[:, 2] ** 2)
            verts /= torch.sqrt(max_len)

            # get neighbors
            faces_contain_this_vertex = []
            for i in range(len(verts)):
                faces_contain_this_vertex.append(set([]))
            for i in range(len(faces)):
                [v1, v2, v3] = faces[i]
                faces_contain_this_vertex[v1].add(i)
                faces_contain_this_vertex[v2].add(i)
                faces_contain_this_vertex[v3].add(i)
            neighbors = []
            for i in range(len(faces)):
                [v1, v2, v3] = faces[i]
                n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
                n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
                n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
                neighbors.append([n1, n2, n3])
            neighbors = np.array(neighbors)


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
            
            try:
                np.asarray(faces_neighbor_1st_ring)
            except:
                print("{} failed".format(path))
                continue

            # paths_dataset.append(path)

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

            # get corners
            corners = verts[faces.long()]
            # 每个面在1st Ring中连接到3个其他面
            assert corners.shape == (max_faces, 3, 3)

            centers = torch.sum(corners, axis=1)/3
            assert centers.shape == (max_faces, 3)

            corners = corners.reshape(-1, 9)
            
            assert f_normals.shape == (max_faces, 3)

            # faces_feature = np.concatenate([centers, corners, f_normals], axis=1)
            # assert faces_feature.shape == (max_faces, 15)

            # 计算相对于数据根目录的相对路径
            rel_path = os.path.relpath(path, data_root)
            # 构建输出文件路径
            output_path = os.path.join(output_root, rel_path.replace('.obj', '.npz'))
            # 创建输出目录
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            label = get_manifold40_label(path)

            max_ver = 252
            verts = np.pad(verts, ((0, max_ver - verts.shape[0]), (0, 0)), mode='constant')
            # verts = torch.from_numpy(verts)


            # 保存NPZ文件
            np.savez(output_path,
                    verts=verts,
                    faces=faces,
                    f_normals=f_normals,
                    v_normals=v_normals,
                    neighbors=neighbors,
                    corners=corners,
                    centers=centers,
                    ring_1=faces_neighbor_1st_ring,
                    ring_2=faces_neighbor_2nd_ring,
                    ring_3=faces_neighbor_3rd_ring,
                    label = label)
            
            print(f'已成功处理: {path}')
            print(f'输出文件: {output_path}')
            
        except Exception as e:
            print(f'处理 {path} 时出错: {e}')
            continue

    print('所有文件处理完成！')


if __name__ == "__main__":
    main()

