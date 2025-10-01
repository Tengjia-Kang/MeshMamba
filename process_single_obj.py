"""
处理单个OBJ文件的脚本，用于计算mesh的neighbors、ring_1、ring_2、ring_3等数据

用法:
  python process_single_obj.py --input path/to/model.obj --output path/to/output.npz
"""
import os
import argparse
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh
from trimesh.graph import face_adjacency
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes


def is_mesh_valid(mesh):
    """
    检查pytorch3D网格的有效性
    """
    validity = True

    # 检查网格是否为空
    if mesh.isempty():
        validity = False

    # 检查网格中的顶点是否有效
    verts = mesh.verts_packed()
    if not torch.isfinite(verts).all() or torch.isnan(verts).all():
        validity = False

    # 检查网格中的顶点法线是否有效
    v_normals = mesh.verts_normals_packed()
    if not torch.isfinite(v_normals).all() or torch.isnan(v_normals).all():
        validity = False

    # 检查网格中的面法线是否有效
    f_normals = mesh.faces_normals_packed()
    if not torch.isfinite(f_normals).all() or torch.isnan(f_normals).all():
        validity = False

    return validity


def find_neighbor(faces, faces_contain_this_vertex, v1, v2, face_idx):
    """
    找到共享边v1-v2的相邻面索引
    """
    intersect = faces_contain_this_vertex[v1].intersection(faces_contain_this_vertex[v2])
    intersect.discard(face_idx)
    if len(intersect) == 0:
        return face_idx  # 如果没有邻居，返回自身索引
    else:
        return intersect.pop()


def process_single_obj(input_path, output_path=None, device='cpu', normalize=True):
    """
    处理单个OBJ文件，计算环形邻居并保存结果
    
    参数:
        input_path: OBJ文件的路径
        output_path: 输出NPZ文件的路径，如果为None则在输入目录下生成
        device: 计算设备('cpu'或'cuda')
        normalize: 是否对网格进行归一化
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
        
    # 检查文件格式是否为OBJ
    if not input_path.lower().endswith('.obj'):
        raise ValueError(f"输入文件必须是OBJ格式: {input_path}")
    
    # 设置默认输出路径
    if output_path is None:
        base_dir = os.path.dirname(input_path)
        base_name = os.path.basename(input_path).replace('.obj', '.npz')
        output_path = os.path.join(base_dir, base_name)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"正在处理文件: {input_path}")
    print(f"输出将保存到: {output_path}")
    
    # 加载OBJ文件
    mesh = load_objs_as_meshes([input_path], device=device)
    faces = mesh.faces_packed()
    verts = mesh.verts_packed()
    edges = mesh.edges_packed()
    
    # 检查网格有效性
    if not is_mesh_valid(mesh):
        print(f'警告: 网格 {input_path} 无效！')
        return False
    
    # 移动到中心并归一化
    if normalize:
        # 移动到中心
        center = (torch.max(verts, 0)[0] + torch.min(verts, 0)[0]) / 2
        verts -= center
        
        # 归一化
        max_len = torch.max(verts[:, 0] ** 2 + verts[:, 1] ** 2 + verts[:, 2] ** 2)
        verts /= torch.sqrt(max_len)
        
        # 更新mesh对象
        mesh = Meshes(verts=[verts], faces=[faces])
        faces = mesh.faces_packed()
        verts = mesh.verts_packed()
        edges = mesh.edges_packed()
    
    # 计算邻居
    print("正在计算邻居关系...")
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
    
    # 计算1st-Ring邻居
    print("正在计算1st-Ring邻居...")
    data = Data(pos=verts, edge_index=edges.permute(1, 0), face=faces.permute(1, 0))
    trimesh_mesh = to_trimesh(data)
    
    # 计算面邻接关系
    faces_adjacency, edges_adjacency = face_adjacency(faces=faces.permute(1, 0),
                                                     mesh=trimesh_mesh,
                                                     return_edges=True)
    
    faces_neighbor_1st_ring = []
    for face_idx in range(len(faces)):
        face_dim_0 = np.argwhere(faces_adjacency[:, 0] == face_idx)
        face_dim_1 = np.argwhere(faces_adjacency[:, 1] == face_idx)
        
        face_neighbor_dim_0 = faces_adjacency[:, 0][face_dim_1]
        face_neighbor_dim_1 = faces_adjacency[:, 1][face_dim_0]
        
        face_neighbor_1st_ring_face = np.concatenate([face_neighbor_dim_0, face_neighbor_dim_1])
        faces_neighbor_1st_ring.append(face_neighbor_1st_ring_face)
    
    faces_neighbor_1st_ring = np.asarray(faces_neighbor_1st_ring)
    
    # 处理形状不一致的情况
    if len(faces_neighbor_1st_ring.shape) > 2:
        faces_neighbor_1st_ring = faces_neighbor_1st_ring.squeeze(2)
    
    # 确保每个面在1st-Ring中有3个邻居
    if faces_neighbor_1st_ring.shape[1] != 3:
        print(f"警告: 1st-Ring邻居形状不正确: {faces_neighbor_1st_ring.shape}")
    
    # 计算2nd-Ring邻居
    print("正在计算2nd-Ring邻居...")
    try:
        if len(faces_neighbor_1st_ring.shape) == 1:
            faces_neighbor_1st_ring = faces_neighbor_1st_ring.reshape(-1, 1)
        
        faces_neighbor_2nd_ring = []
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
            faces_neighbor_2nd_ring.append(neighbors_2nd)
        
        faces_neighbor_2nd_ring = np.array(faces_neighbor_2nd_ring)
    except Exception as e:
        print(f"计算2nd-Ring邻居时出错: {e}")
        # 创建默认的2nd-Ring邻居
        faces_neighbor_2nd_ring = np.tile(np.arange(6) % len(faces), (len(faces), 1))
    
    # 计算3rd-Ring邻居
    print("正在计算3rd-Ring邻居...")
    try:
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
        print(f"计算3rd-Ring邻居时出错: {e}")
        # 创建默认的3rd-Ring邻居
        faces_neighbor_3rd_ring = np.tile(np.arange(12) % len(faces), (len(faces), 1))
    
    # 保存数据
    print(f"正在保存结果到: {output_path}")
    np.savez(output_path,
             faces=faces.cpu().numpy(),
             verts=verts.cpu().numpy(),
             ring_1=faces_neighbor_1st_ring,
             ring_2=faces_neighbor_2nd_ring,
             ring_3=faces_neighbor_3rd_ring,
             neighbors=neighbors)
    
    print("处理完成！")
    return True


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理单个OBJ文件，计算mesh的环形邻居')
    parser.add_argument('--input', '-i', required=True, help='输入OBJ文件的路径')
    parser.add_argument('--output', '-o', help='输出NPZ文件的路径')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='计算设备')
    parser.add_argument('--no-normalize', action='store_true', help='不对网格进行归一化')
    
    args = parser.parse_args()
    
    # 如果指定了cuda但不可用，则回退到cpu
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，将使用CPU")
        args.device = 'cpu'
    
    # 处理OBJ文件
    process_single_obj(
        input_path=args.input,
        output_path=args.output,
        device=args.device,
        normalize=not args.no_normalize
    )


if __name__ == "__main__":
    main()