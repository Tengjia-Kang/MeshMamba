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

    # trimesh load uv and texture
    # transform = transforms.ToTensor()
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    # ])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(1024),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    vertices, faces_idx, aux = load_obj(f_path)
    grid = 2.0 * aux.verts_uvs[faces_idx.textures_idx] - 1
    grid = grid.unsqueeze(0)
    img = aux.texture_images['material_0'].numpy()
    img_tensor = transform(img).unsqueeze(0).flip(2)

    # Grid Sample face colors
    face_colors = F.grid_sample(img_tensor, grid=grid, mode='nearest', align_corners=True)
    face_colors = face_colors.squeeze().permute(1, 2, 0)

    # Grid Sample Texture Patches
    grid_size = 9
    x_min = torch.min(grid[:, :, :, 0], dim=2)[0]
    x_max = torch.max(grid[:, :, :, 0], dim=2)[0]
    y_min = torch.min(grid[:, :, :, 1], dim=2)[0]
    y_max = torch.max(grid[:, :, :, 1], dim=2)[0]
    step_x = (x_max - x_min) / (grid_size - 1)
    step_y = (y_max - y_min) / (grid_size - 1)
    indices = torch.linspace(0, grid_size - 1, grid_size)

    # get min/max step and min_edge
    aspect_ratio = 'keep_aspect_ratio'  # 'keep_aspect_ratio' or 'resize_to_square'
    if aspect_ratio == 'keep_aspect_ratio':
        min_max = torch.cat([x_max - x_min, y_max - y_min], dim=0).min(0)[1]
        x_min[0, min_max == 0] = x_min[0, min_max == 0] - 0.5 * ((y_max - y_min) - (x_max - x_min))[0, min_max == 0]
        y_min[0, min_max == 1] = y_min[0, min_max == 1] + 0.5 * ((y_max - y_min) - (x_max - x_min))[0, min_max == 1]
        step_x[0, min_max == 0] = step_y[0, min_max == 0]
        step_y[0, min_max == 1] = step_x[0, min_max == 1]

    xx_linspaces = x_min[:, :, None] + indices * step_x[:, :, None]
    yy_linspaces = y_min[:, :, None] + indices * step_y[:, :, None]
    grid_x = xx_linspaces.unsqueeze(3).expand(-1, -1, -1, grid_size)
    grid_y = yy_linspaces.unsqueeze(2).expand(-1, -1, grid_size, -1)
    grid_mean = torch.stack([grid_x, grid_y], dim=-1)
    # at 1
    face_textures1 = F.grid_sample(img_tensor.expand(32868, -1, -1, -1),
                                   grid=grid_mean.squeeze(), mode='nearest', align_corners=True)
    # at 2
    img_tensor_ = img_tensor.unsqueeze(2)
    grid_mean_ = torch.cat([grid_mean, torch.zeros_like(grid_mean)[:,:,:,:,0].unsqueeze(-1)], -1)
    face_textures2 = F.grid_sample(img_tensor_, grid=grid_mean_, mode='nearest', align_corners=True)
    face_textures2 = face_textures2.squeeze().transpose(0, 1)

    face_textures = face_textures1
    # face_colors = face_textures[:, :, 0, 0]
    # face_colors = (face_colors - torch.min(face_colors)) / (torch.max(face_colors) - torch.min(face_colors))
    # face_colors = face_textures.reshape((face_textures.shape[0], face_textures.shape[1], -1))
    # face_colors = torch.mean(face_colors, 2)
    # import trimesh
    # mesh_new = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=face_colors)
    # mesh_new.visual.face_colors = face_colors
    # mesh_new.export("aaa.obj")

    # save texture and uv
    texture = img_tensor.squeeze()
    uv_grid = grid_mean_.squeeze()
    # print(texture.shape)
    return (mesh, faces, verts, edges, v_normals, f_normals,
            face_colors, face_textures, texture, uv_grid)




def main():
    device = torch.device('cuda:0')
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
                    f_normals=f_normals,
                    v_normals=v_normals,
                    neighbors=neighbors,
                    corners=corners,
                    centers=centers,
                    ring_1=faces_neighbor_1st_ring,
                    ring_2=faces_neighbor_2nd_ring,
                    ring_3=faces_neighbor_3rd_ring)
            
            print(f'已成功处理: {path}')
            print(f'输出文件: {output_path}')
            
        except Exception as e:
            print(f'处理 {path} 时出错: {e}')
            continue

    print('所有文件处理完成！')


if __name__ == "__main__":
    main()

