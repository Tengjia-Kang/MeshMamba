"""
Data is pre-processed to obtain the following infomation for non-textured meshes:
    1) Vertices and Faces of the mesh
    2) 1 Ring, 2 Ring, and 3 Ring neighborhood of the mesh faces
"""
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh
from trimesh.graph import face_adjacency
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes


def is_mesh_valid(mesh):
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


def pytorch3D_mesh(f_path, device):
    """
    Read pytorch3D mesh from path for non-textured meshes

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


def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face


def main():
    device = torch.device('cpu')
    # dataset base root
    data_root = '/home/kangkang/Projects/MeshMamba/dataset/Manifold40'
    # To process the dataset enter the path where they are stored
    save_dir = '/home/kangkang/Projects/MeshMamba/dataset/Manifold40_ringn'  # 处理无纹理的mesh
    max_faces = 500
    if not os.path.exists(data_root):
        raise Exception('Dataset not found at {0}'.format(data_root))

    paths_dataset = []
    fpath_data = fpath(data_root)
    fpath_data.sort()
    for path in fpath_data:
        print(f"Processing: {path}")
        
        # 调用无纹理版本的mesh加载函数
        (mesh, faces, verts, edges, v_normals, f_normals) = pytorch3D_mesh(path, device)
        
        current_faces = faces.shape[0]
        if current_faces > max_faces:
        # 面数过多，随机采样500个面
            indices = torch.randperm(current_faces)[:max_faces]
            faces = faces[indices]
            f_normals = f_normals[indices] if f_normals is not None else None
        elif current_faces < max_faces:
        # 面数不足，重复填充到500个面
            repeat_times = (max_faces // current_faces) + 1
            faces = faces.repeat(repeat_times, 1)[:max_faces]
            if f_normals is not None:
                f_normals = f_normals.repeat(repeat_times, 1)[:max_faces]
        
        # 检查mesh有效性
        if not is_mesh_valid(mesh):
            print(f'Warning: Mesh {path} is invalid, skipping!')
            continue
        
        # 移动到中心
        center = (torch.max(verts, 0)[0] + torch.min(verts, 0)[0]) / 2
        verts -= center
        
        # 归一化
        max_len = torch.max(verts[:, 0] ** 2 + verts[:, 1] ** 2 + verts[:, 2] ** 2)
        verts /= torch.sqrt(max_len)
        
        # 计算邻居
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
        # Neighbor faces index along edges, Edges along neighbor_faces
        faces_adjacency, edges_adjacency = face_adjacency(faces=faces.permute(1, 0),
                                                          mesh=trimesh,
                                                          return_edges=True)
        
        faces_neighbor_1st_ring = []
        edges_neighbor_1ring = []
        
        # For each face get 1-Ring neighborhood along its edges
        # For each face get edge between face and neighbor faces
        for face_idx in range(len(faces)):
            face_dim_0 = np.argwhere(faces_adjacency[:, 0] == face_idx)
            face_dim_1 = np.argwhere(faces_adjacency[:, 1] == face_idx)
            
            face_neighbor_dim_0 = faces_adjacency[:, 0][face_dim_1]
            face_neighbor_dim_1 = faces_adjacency[:, 1][face_dim_0]
            
            face_neighbor_1st_ring_face = np.concatenate([face_neighbor_dim_0, face_neighbor_dim_1])
            
            # Edge between face and neighbor faces
            face_edge = np.concatenate([face_dim_0, face_dim_1]).reshape(-1)
            edge_neighbor_1ring_face = edges_adjacency[face_edge]
            
            faces_neighbor_1st_ring.append(face_neighbor_1st_ring_face)
            edges_neighbor_1ring.append(edge_neighbor_1ring_face)
        
        try:
            np.asarray(faces_neighbor_1st_ring)
        except:
            print(f"Error processing {path}")
            continue
        
        paths_dataset.append(path)
        faces_neighbor_1st_ring = np.asarray(faces_neighbor_1st_ring)
        edges_neighbor_1ring = np.asarray(edges_neighbor_1ring)
        
        # 处理形状不一致的情况
        if len(faces_neighbor_1st_ring.shape) > 2:
            faces_neighbor_1st_ring = faces_neighbor_1st_ring.squeeze(2)
        
        # 确保每个面在1st-Ring中有3个邻居
        if faces_neighbor_1st_ring.shape[1] != 3:
            print(f"Warning: {path} has incorrect 1st ring neighbors shape: {faces_neighbor_1st_ring.shape}")
        
        ########################################################################### 2nd-Ring ###########################################################################
        faces_neighbor_0th_ring = np.arange(len(faces))
        
        # 处理形状以确保正确索引
        try:
            if len(faces_neighbor_1st_ring.shape) == 1:
                faces_neighbor_1st_ring = faces_neighbor_1st_ring.reshape(-1, 1)
            
            # 计算2nd-Ring邻居
            faces_neighbor_2ring = []
            for i in range(len(faces)):
                neighbors_2nd = []
                for neighbor_idx in faces_neighbor_1st_ring[i]:
                    if neighbor_idx < len(faces):  # 确保索引有效
                        neighbors_2nd.extend(faces_neighbor_1st_ring[neighbor_idx])
                # 移除重复项和自身
                neighbors_2nd = list(set(neighbors_2nd) - {i} - set(faces_neighbor_1st_ring[i]))
                # 填充或截断到6个邻居
                if len(neighbors_2nd) < 6:
                    neighbors_2nd += [i] * (6 - len(neighbors_2nd))
                else:
                    neighbors_2nd = neighbors_2nd[:6]
                faces_neighbor_2ring.append(neighbors_2nd)
            
            faces_neighbor_2nd_ring = np.array(faces_neighbor_2ring)
        except Exception as e:
            print(f"Error calculating 2nd ring for {path}: {e}")
            # 创建默认的2nd-Ring邻居
            faces_neighbor_2nd_ring = np.tile(np.arange(6) % len(faces), (len(faces), 1))
        
        ########################################################################### 3rd-Ring ###########################################################################
        try:
            # 计算3rd-Ring邻居
            faces_neighbor_3rd_ring = []
            for i in range(len(faces)):
                neighbors_3rd = []
                for neighbor_idx in faces_neighbor_2nd_ring[i]:
                    if neighbor_idx < len(faces):  # 确保索引有效
                        neighbors_3rd.extend(faces_neighbor_1st_ring[neighbor_idx])
                # 移除重复项、自身和1st-Ring邻居
                exclude_set = {i} | set(faces_neighbor_1st_ring[i]) | set(faces_neighbor_2nd_ring[i])
                neighbors_3rd = list(set(neighbors_3rd) - exclude_set)
                # 填充或截断到12个邻居
                if len(neighbors_3rd) < 12:
                    neighbors_3rd += [i] * (12 - len(neighbors_3rd))
                else:
                    neighbors_3rd = neighbors_3rd[:12]
                faces_neighbor_3rd_ring.append(neighbors_3rd)
            
            faces_neighbor_3rd_ring = np.array(faces_neighbor_3rd_ring)
        except Exception as e:
            print(f"Error calculating 3rd ring for {path}: {e}")
            # 创建默认的3rd-Ring邻居
            faces_neighbor_3rd_ring = np.tile(np.arange(12) % len(faces), (len(faces), 1))
        

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 构建保存路径
        filename = os.path.basename(path).replace('.obj', '.npz')
        path_ring = os.path.join(save_dir, filename)
        
        # 保存数据
        np.savez(path_ring,
                 faces=faces,
                 ring_1=faces_neighbor_1st_ring,
                 ring_2=faces_neighbor_2nd_ring,
                 ring_3=faces_neighbor_3rd_ring,
                 neighbors=neighbors)
        
        print(f"Saved: {path_ring}")
    
    # 保存处理过的文件路径列表
    np.savetxt("paths_dataset_non_texture.txt", paths_dataset, fmt="%s", delimiter=",")
    print(f"Total processed meshes: {len(paths_dataset)}")


if __name__ == "__main__":
    main()