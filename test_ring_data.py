"""
Test script to demonstrate how to load and use the preprocessed ring neighborhood data for non-textured meshes.
"""
import numpy as np
import os


def load_ring_data(file_path):
    """
    Load the preprocessed ring neighborhood data from npz file.
    
    Args:
        file_path: Path to the npz file containing the preprocessed data.
    
    Returns:
        data: Dictionary containing the loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    data = np.load(file_path)
    return data


def visualize_ring_neighborhood(faces, ring_1, ring_2, ring_3, face_idx=0):
    """
    Visualize the ring neighborhood for a specific face.
    
    Args:
        faces: Array of face indices.
        ring_1: 1-Ring neighborhood indices.
        ring_2: 2-Ring neighborhood indices.
        ring_3: 3-Ring neighborhood indices.
        face_idx: Index of the face to visualize.
    """
    print(f"\n=== Ring Neighborhood for Face {face_idx} ===")
    print(f"Face {face_idx} vertices: {faces[face_idx]}")
    print(f"1-Ring neighbors ({len(ring_1[face_idx])}): {ring_1[face_idx]}")
    print(f"2-Ring neighbors ({len(ring_2[face_idx])}): {ring_2[face_idx]}")
    print(f"3-Ring neighbors ({len(ring_3[face_idx])}): {ring_3[face_idx]}")
    
    # 验证邻居索引的合理性
    max_face_idx = len(faces)
    valid_1ring = all(0 <= idx < max_face_idx for idx in ring_1[face_idx])
    valid_2ring = all(0 <= idx < max_face_idx for idx in ring_2[face_idx])
    valid_3ring = all(0 <= idx < max_face_idx for idx in ring_3[face_idx])
    
    print(f"\nValidation:")
    print(f"1-Ring indices valid: {valid_1ring}")
    print(f"2-Ring indices valid: {valid_2ring}")
    print(f"3-Ring indices valid: {valid_3ring}")


def check_ring_hierarchy(ring_1, ring_2, ring_3, face_idx=0):
    """
    Check the hierarchy of ring neighborhoods.
    
    Args:
        ring_1: 1-Ring neighborhood indices.
        ring_2: 2-Ring neighborhood indices.
        ring_3: 3-Ring neighborhood indices.
        face_idx: Index of the face to check.
    """
    # 检查1-Ring和2-Ring是否有重叠（应该没有）
    ring_1_set = set(ring_1[face_idx])
    ring_2_set = set(ring_2[face_idx])
    overlap_1_2 = ring_1_set.intersection(ring_2_set)
    
    # 检查2-Ring和3-Ring是否有重叠（应该没有）
    ring_3_set = set(ring_3[face_idx])
    overlap_2_3 = ring_2_set.intersection(ring_3_set)
    
    # 检查1-Ring和3-Ring是否有重叠（应该没有）
    overlap_1_3 = ring_1_set.intersection(ring_3_set)
    
    print(f"\n=== Ring Hierarchy Check ===")
    print(f"Overlap between 1-Ring and 2-Ring: {len(overlap_1_2)} faces")
    print(f"Overlap between 2-Ring and 3-Ring: {len(overlap_2_3)} faces")
    print(f"Overlap between 1-Ring and 3-Ring: {len(overlap_1_3)} faces")


def main():
    # 设置数据文件路径
    data_dir = './dataset/Ring/non_texture'
    sample_file = os.path.join(data_dir, 'Apple_v01_l3.npz')
    
    # 加载数据
    print(f"Loading data from: {sample_file}")
    data = load_ring_data(sample_file)
    
    # 打印数据信息
    print(f"Loaded data keys: {list(data.keys())}")
    print(f"Number of faces: {len(data['faces'])}")
    print(f"1-Ring shape: {data['ring_1'].shape}")
    print(f"2-Ring shape: {data['ring_2'].shape}")
    print(f"3-Ring shape: {data['ring_3'].shape}")
    print(f"Neighbors shape: {data['neighbors'].shape}")
    
    # 可视化几个面的环形邻居
    face_indices = [0, 100, 1000]
    for face_idx in face_indices:
        visualize_ring_neighborhood(
            data['faces'], 
            data['ring_1'], 
            data['ring_2'], 
            data['ring_3'], 
            face_idx
        )
        
        # 检查环形邻居的层次结构
        check_ring_hierarchy(
            data['ring_1'], 
            data['ring_2'], 
            data['ring_3'], 
            face_idx
        )


if __name__ == "__main__":
    main()