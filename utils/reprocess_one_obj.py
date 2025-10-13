# utils/reprocess_one_obj.py
import os
import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from torch_geometric.data import Data
from torch_geometric.utils import to_trimesh
from trimesh.graph import face_adjacency


"""
    Reprocess one obj file to make sure it has max_faces faces.
    If it has more than max_faces faces, randomly sample max_faces faces.
    If it has less than max_faces faces, repeat faces until reach max_faces.
"""
def reprocess_obj(obj_path, save_dir, max_faces=500, device='cpu'):
    print("reprocessing:", obj_path)
    mesh = load_objs_as_meshes([obj_path], device=device)
    faces = mesh.faces_packed().cpu().numpy()  # (F,3)
    verts = mesh.verts_packed().cpu().numpy()  # (V,3)

    F_cnt = faces.shape[0]
    if F_cnt > max_faces:
        idx = np.random.permutation(F_cnt)[:max_faces]
        faces = faces[idx]
    elif F_cnt < max_faces:
        # repeat faces until reach max_faces
        repeats = (max_faces + F_cnt - 1) // F_cnt
        faces = np.tile(faces, (repeats, 1))[:max_faces]

    # clamp indices to valid vertex range just in case
    faces = np.clip(faces, 0, verts.shape[0]-1).astype(np.int64)

    # build trimesh helper to compute adjacency
    try:
        data = Data(pos=torch.from_numpy(verts), edge_index=mesh.edges_packed().permute(1,0), face=torch.from_numpy(faces).permute(1,0))
        trimesh_obj = to_trimesh(data)
        faces_adj, edges_adj = face_adjacency(faces=faces.T, mesh=trimesh_obj, return_edges=True)

        # build 1st ring
        faces_neighbor_1st = []
        for face_idx in range(len(faces)):
            face_dim_0 = np.argwhere(faces_adj[:,0] == face_idx)
            face_dim_1 = np.argwhere(faces_adj[:,1] == face_idx)
            face_neighbor_dim_0 = faces_adj[:,0][face_dim_1].reshape(-1)
            face_neighbor_dim_1 = faces_adj[:,1][face_dim_0].reshape(-1)
            neigh = np.concatenate([face_neighbor_dim_0, face_neighbor_dim_1])
            # remove duplicates and keep length 3
            neigh = [int(x) for x in neigh if int(x) != face_idx]
            neigh = list(dict.fromkeys(neigh))
            if len(neigh) < 3:
                neigh += [face_idx] * (3 - len(neigh))
            faces_neighbor_1st.append(neigh[:3])
        ring1 = np.array(faces_neighbor_1st, dtype=np.int64)

        # 2nd ring (build using ring1)
        ring2 = []
        for i in range(len(faces)):
            s = []
            for nb in ring1[i]:
                s.extend(list(ring1[nb]))
            s = [int(x) for x in s if int(x) != i and x not in ring1[i]]
            s = list(dict.fromkeys(s))
            if len(s) < 6:
                s += [i] * (6 - len(s))
            ring2.append(s[:6])
        ring2 = np.array(ring2, dtype=np.int64)

        # 3rd ring
        ring3 = []
        for i in range(len(faces)):
            s = []
            for nb in ring2[i]:
                s.extend(list(ring1[nb]))
            exclude = set([i]) | set(ring1[i]) | set(ring2[i])
            s = [int(x) for x in s if int(x) not in exclude]
            s = list(dict.fromkeys(s))
            if len(s) < 12:
                s += [i] * (12 - len(s))
            ring3.append(s[:12])
        ring3 = np.array(ring3, dtype=np.int64)

        # compute neighbors from faces (shared edge neighbor)
        neighbors = []
        # naive neighbor: for each face, pick neighbor indices from ring1 (already 3)
        neighbors = ring1.copy()

    except Exception as e:
        print("adjacency compute failed:", e)
        # fallback zero arrays
        ring1 = np.tile(np.arange(3) % len(faces), (len(faces), 1))
        ring2 = np.tile(np.arange(6) % len(faces), (len(faces), 1))
        ring3 = np.tile(np.arange(12) % len(faces), (len(faces), 1))
        neighbors = np.tile(np.arange(3) % len(faces), (len(faces), 1))

    # save
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fn = os.path.basename(obj_path).replace('.obj', '.npz')
    save_path = os.path.join(save_dir, fn)
    np.savez(save_path, faces=faces.astype(np.int64), ring_1=ring1, ring_2=ring2, ring_3=ring3, neighbors=neighbors)
    print("saved:", save_path)
    return save_path

if __name__ == "__main__":
    # example usage: 填入实际路径并运行
    obj = "/mnt/newdisk/ktj/Mesh/Manifold40/bookshelf/train/bookshelf_0099.obj"
    save_dir = "/home/ktj/Projects/MeshMamba/dataset/processed/Manifold_ringn"
    reprocess_obj(obj, save_dir, max_faces=500, device='cpu')
