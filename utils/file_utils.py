"""
Collection of utility function for reading file in batches
"""
import os
import numpy as np
import subprocess

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







def get_dataset_paths(data_root):
    """
    Get all obj file paths in train and test folders of Manifold40 dataset.

    Args:
        data_root: root path to Manifold40 dataset

    Returns:
        dataset_paths: dictionary with 'train' and 'test' keys, each containing a list of obj file paths
    """
    # 初始化两个列表来存储文件路径
    train_files = []
    test_files = []

    # 遍历40个类别文件夹
    for category in os.listdir(data_root):
        category_path = os.path.join(data_root, category)
        
        # 检查是否是目录
        if os.path.isdir(category_path):
            # 处理train文件夹
            train_dir = os.path.join(category_path, 'train')
            if os.path.exists(train_dir) and os.path.isdir(train_dir):
                # 遍历train文件夹下的所有.obj文件
                for file in os.listdir(train_dir):
                    if file.endswith('.obj'):
                        file_path = os.path.join(train_dir, file)
                        train_files.append(file_path)
            
            # 处理test文件夹
            test_dir = os.path.join(category_path, 'test')
            if os.path.exists(test_dir) and os.path.isdir(test_dir):
                # 遍历test文件夹下的所有.obj文件
                for file in os.listdir(test_dir):
                    if file.endswith('.obj'):
                        file_path = os.path.join(test_dir, file)
                        test_files.append(file_path)

    
    # 现在train_files包含了所有类别的train文件夹中的.obj文件路径
    # test_files包含了所有类别的test文件夹中的.obj文件路径
    print(f"找到{len(train_files)}个训练文件")
    print(f"找到{len(test_files)}个测试文件")
    
    # 返回包含训练和测试文件路径的字典
    return {'train': train_files, 'test': test_files}