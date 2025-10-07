import torch
import yaml
import os
from models.MambaMesh_Modified import MambaMeshForClassification

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def debug_model():
    # 加载配置文件
    config_path = '/home/ktj/Projects/MeshMamba/config/Manifold40.yaml'
    print(f"加载配置文件: {config_path}")
    config = load_config(config_path)
    
    # 提取mamba配置
    mamba_config = config['mamba']
    print("Mamba配置参数:")
    for key, value in mamba_config.items():
        print(f"  {key}: {value}")
    
    # 创建模型实例
    print("\n创建模型实例...")
    model = MambaMeshForClassification(mamba_config)
    
    # 检查是否有GPU可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)
    
    # 创建模拟输入数据
    batch_size = 2
    num_points = 4096
    
    # 模拟必需的输入参数
    # 顶点数据 (B, 3, N)
    verts = torch.randn(batch_size, 3, num_points).to(device)
    # 面数据 (B, 3, F) - F是面的数量
    num_faces = num_points // 2  # 假设每个顶点属于两个面
    faces = torch.randint(0, num_points, (batch_size, 3, num_faces)).to(device)
    # 中心点数据 (B, 3, N)
    centers = torch.randn(batch_size, 3, num_points).to(device)
    # 法线数据 (B, 3, N)
    normals = torch.randn(batch_size, 3, num_points).to(device)
    # 顶点坐标数据 (B, 9, N) - 每个面3个顶点，每个顶点3个坐标
    corners = torch.randn(batch_size, 9, num_points).to(device)
    # 查看FaceShape_Extractor和Structural_Extractor中的forward方法实现
    # 从错误信息看，维度顺序应该是[B, N, K]而不是[B, K, N]
    num_neighbor = 3
    
    # 邻居索引 (B, N, K) - K是每个点的邻居数量，维度顺序要与FaceShape_Extractor的期望匹配
    neighbor_index = torch.randint(0, num_points, (batch_size, num_points, num_neighbor)).to(device)
    # 环形邻居数据 (B, N, K) - 维度顺序必须是[B, N, K]以匹配FaceShape_Extractor中的gather操作
    ring_1 = torch.randint(0, num_points, (batch_size, num_points, num_neighbor)).to(device)
    ring_2 = torch.randint(0, num_points, (batch_size, num_points, num_neighbor)).to(device)
    ring_3 = torch.randint(0, num_points, (batch_size, num_points, num_neighbor)).to(device)
    
    print(f"\n模拟输入形状:")
    print(f"  verts: {verts.shape}")
    print(f"  faces: {faces.shape}")
    print(f"  centers: {centers.shape}")
    print(f"  normals: {normals.shape}")
    print(f"  corners: {corners.shape}")
    print(f"  neighbor_index: {neighbor_index.shape}")
    print(f"  ring_1: {ring_1.shape}")
    print(f"  ring_2: {ring_2.shape}")
    print(f"  ring_3: {ring_3.shape}")
    
    # 尝试前向传播
    try:
        print("\n尝试模型前向传播...")
        # 传递所有必需的参数给forward方法
        output = model(verts, faces, centers, normals, corners, neighbor_index,
                      ring_1, ring_2, ring_3)
        print(f"前向传播成功!")
        print(f"输出形状: {output.shape}")
        
        # 如果成功，进行一次反向传播测试
        print("\n尝试反向传播...")
        output.sum().backward()
        print("反向传播成功!")
        
        return True
    except Exception as e:
        print(f"\n前向传播失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== MambaMesh模型调试工具 ===")
    success = debug_model()
    print(f"\n调试 {'成功' if success else '失败'}")