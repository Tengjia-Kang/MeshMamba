#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本：验证preprocess.py是否正确支持指定输出目录并保持子文件夹结构
"""
import os
import shutil
import subprocess
import numpy as np
import tempfile

# 临时目录路径
test_dir = tempfile.mkdtemp(prefix='mesh_test_')
input_dir = os.path.join(test_dir, 'input')
output_dir = os.path.join(test_dir, 'output')

print(f'创建临时测试目录: {test_dir}')

# 创建测试目录结构和示例OBJ文件
def create_test_structure():
    # 创建输入目录结构
    os.makedirs(os.path.join(input_dir, 'category1', 'train'), exist_ok=True)
    os.makedirs(os.path.join(input_dir, 'category1', 'test'), exist_ok=True)
    os.makedirs(os.path.join(input_dir, 'category2', 'train'), exist_ok=True)
    
    # 创建简单的立方体OBJ文件（8个顶点，12个面）
    cube_obj_content = """
v 1.0 1.0 -1.0
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
    
    # 创建多个测试OBJ文件
    obj_files = [
        os.path.join(input_dir, 'category1', 'train', 'cube1.obj'),
        os.path.join(input_dir, 'category1', 'test', 'cube2.obj'),
        os.path.join(input_dir, 'category2', 'train', 'cube3.obj')
    ]
    
    for obj_file in obj_files:
        with open(obj_file, 'w') as f:
            f.write(cube_obj_content)
        print(f'已创建测试文件: {obj_file}')

# 运行preprocess.py进行测试
def run_preprocess():
    # 运行preprocess.py命令，指定输入和输出目录
    cmd = [
        'python', 'preprocess.py',
        '--input_dir', input_dir,
        '--output_dir', output_dir,
        '--max_faces', '12'
    ]
    
    print(f'运行命令: {" ".join(cmd)}')
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 打印输出结果
    print('\n=== 命令输出 ===')
    print(result.stdout)
    print(result.stderr)
    
    # 检查命令是否成功执行
    if result.returncode != 0:
        print(f'命令执行失败，返回码: {result.returncode}')
        return False
    
    return True

# 验证输出结果
def verify_output():
    # 检查输出目录结构
    print('\n=== 验证输出结果 ===')
    
    # 预期的NPZ文件路径
    expected_npz_files = [
        os.path.join(output_dir, 'category1', 'train', 'cube1.npz'),
        os.path.join(output_dir, 'category1', 'test', 'cube2.npz'),
        os.path.join(output_dir, 'category2', 'train', 'cube3.npz')
    ]
    
    # 检查每个文件是否存在并包含正确的数据
    all_valid = True
    
    for npz_file in expected_npz_files:
        if not os.path.exists(npz_file):
            print(f'错误: 文件不存在 - {npz_file}')
            all_valid = False
            continue
        
        print(f'检查文件: {npz_file}')
        
        # 加载NPZ文件并检查内容
        try:
            data = np.load(npz_file)
            required_keys = ['verts', 'faces', 'ring_1', 'ring_2', 'ring_3']
            
            # 检查必需的键是否存在
            for key in required_keys:
                if key not in data:
                    print(f'错误: {npz_file} 中缺少键 {key}')
                    all_valid = False
                    break
            
            if all(key in data for key in required_keys):
                # 检查数据形状是否正确
                print(f"  - 顶点数: {data['verts'].shape[0]}")
                print(f"  - 面数: {data['faces'].shape[0]}")
                print(f"  - 1-Ring形状: {data['ring_1'].shape}")
                print(f"  - 2-Ring形状: {data['ring_2'].shape}")
                print(f"  - 3-Ring形状: {data['ring_3'].shape}")
        except Exception as e:
            print(f'错误: 加载 {npz_file} 时出错 - {e}')
            all_valid = False
    
    return all_valid

# 清理临时文件
def cleanup():
    if os.path.exists(test_dir):
        print(f'\n清理临时目录: {test_dir}')
        shutil.rmtree(test_dir)

# 主函数
def main():
    try:
        # 创建测试结构
        create_test_structure()
        
        # 运行预处理
        if not run_preprocess():
            print('测试失败: preprocess.py执行失败')
            return
        
        # 验证输出
        if verify_output():
            print('\n测试成功: preprocess.py正确支持指定输出目录并保持子文件夹结构')
        else:
            print('\n测试失败: 输出结果不符合预期')
            
    except Exception as e:
        print(f'测试过程中出错: {e}')
    finally:
        # 清理（取消注释以自动清理）
        # cleanup()
        print('\n测试完成！')

if __name__ == '__main__':
    main()