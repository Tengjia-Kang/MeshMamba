import sys
import os
import yaml
import torch
from data import MeshClassificationDataset

def test_mesh_classification_dataset():
    """测试MeshClassificationDataset类"""
    
    # 配置参数
    cfg = {
        'data_root': '/mnt/newdisk/ktj/Mesh/dataset_preprocessed/Manifold40',
        'max_faces': 500,
        'augment_data': True,
        'augment_vert': False,
        'augment_rotation': True,
        'jitter_sigma': 0.01,
        'jitter_clip': 0.05
    }
    
    print("Testing MeshClassificationDataset...")
    
    try:
        # 创建数据集实例
        dataset = MeshClassificationDataset(cfg, part='train')
        
        print(f"Dataset loaded successfully!")
        print(f"Number of samples: {len(dataset)}")
        print(f"Categories: {dataset.get_categories()}")
        print(f"Number of classes: {dataset.get_num_classes()}")
        
        # 测试获取一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nSample data structure:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: Tensor with shape {value.shape}")
                else:
                    print(f"  {key}: {value}")
                    
        # 测试几个样本
        print("\nTesting multiple samples...")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"Sample {i}: {sample['category']}/{sample['mesh_name']}, label: {sample['label']}")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mesh_classification_dataset()