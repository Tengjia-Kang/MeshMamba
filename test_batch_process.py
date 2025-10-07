#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试批量处理数据集的脚本
"""
import os
import shutil
import subprocess
import tempfile
import numpy as np

def create_test_dataset(input_dir):
    """创建测试数据集"""
    # 创建多层目录结构
    categories = ['airplane', 'chair', 'table']
    splits = ['train', 'test']
    
    # 简单的立方体OBJ内容（8顶点，12面）
    cube_content = """v 1.0 1.0 -1.0
v 1.0 -1.0 -1.0
v -1.0 -1.0 -1.0
v -1.0 1.0 -1.0
v 1.0 1.0 1.0
v 1.0 -1.0 1.0
v -1.0 -1.0 1.0
v -1.0 1.0 1.0
f 1 2 4
f 2 3 4
f 5 8 6
f 8 7 6
f 1 5 2
f 5 6 2
f 2 6 3
f 6 7 3
f 3 7 4
f 7 8 4
f 4 8 1
f 8 5 1
"""
    
    obj_files = []
    for category in categories:
        for split in splits:
            category_dir = os.path.join(input_dir, category, split)
            os.makedirs(category_dir, exist_ok=True)
            
            # 每个类别-分割组合创建2个文件
            for i in range(2):
                obj_file = os.path.join(category_dir, f'{category}_{split}_{i:03d}.obj')
                with open(obj_file, 'w') as f:
                    f.write(cube_content)
                obj_files.append(obj_file)
                print(f'创建测试文件: {obj_file}')
    
    return obj_files

def test_batch_process():
    """测试批量处理功能"""
    # 创建临时目录
    test_dir = tempfile.mkdtemp(prefix='batch_test_')
    input_dir = os.path.join(test_dir, 'input')
    output_dir = os.path.join(test_dir, 'output')
    
    print(f'测试目录: {test_dir}')
    
    try:
        # 创建测试数据集
        obj_files = create_test_dataset(input_dir)
        print(f'总计创建了 {len(obj_files)} 个测试文件')
        
        # 运行批量处理脚本
        cmd = [
            'python', 'batch_process_dataset.py',
            '--input_dir', input_dir,
            '--output_dir', output_dir,
            '--max_faces', '12',
            '--device', 'cpu'
        ]
        
        print(f'运行命令: {" ".join(cmd)}')
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print('\n=== 命令输出 ===')
        print(result.stdout)
        if result.stderr:
            print('=== 错误输出 ===')
            print(result.stderr)
        
        # 检查执行结果
        if result.returncode != 0:
            print(f'命令执行失败，返回码: {result.returncode}')
            return False
        
        # 验证输出结果
        print('\n=== 验证输出结果 ===')
        
        # 检查是否生成了对应的NPZ文件
        expected_npz_files = []
        for obj_file in obj_files:
            rel_path = os.path.relpath(obj_file, input_dir)
            npz_file = os.path.join(output_dir, rel_path.replace('.obj', '.npz'))
            expected_npz_files.append(npz_file)
        
        success_count = 0
        for npz_file in expected_npz_files:
            if not os.path.exists(npz_file):
                print(f'错误: 文件不存在 - {npz_file}')
                continue
            
            try:
                # 加载并验证NPZ文件内容
                data = np.load(npz_file)
                required_keys = ['verts', 'faces', 'ring_1', 'ring_2', 'ring_3', 'neighbors']
                
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    print(f'错误: {npz_file} 缺少键: {missing_keys}')
                    continue
                
                # 检查数据形状
                print(f'文件: {os.path.basename(npz_file)}')
                print(f"  顶点数: {data['verts'].shape[0]}")
                print(f"  面数: {data['faces'].shape[0]}")
                print(f"  1-Ring形状: {data['ring_1'].shape}")
                print(f"  2-Ring形状: {data['ring_2'].shape}")
                print(f"  3-Ring形状: {data['ring_3'].shape}")
                print(f"  邻居数量: {len(data['neighbors'])}")
                
                success_count += 1
                
            except Exception as e:
                print(f'错误: 验证 {npz_file} 时出错 - {e}')
        
        print(f'\n验证结果: {success_count}/{len(expected_npz_files)} 个文件处理成功')
        
        if success_count == len(expected_npz_files):
            print('测试成功: 批量处理脚本正常工作')
            return True
        else:
            print('测试失败: 部分文件处理失败')
            return False
            
    finally:
        # 清理临时文件
        if os.path.exists(test_dir):
            print(f'清理测试目录: {test_dir}')
            shutil.rmtree(test_dir)

if __name__ == '__main__':
    test_batch_process()