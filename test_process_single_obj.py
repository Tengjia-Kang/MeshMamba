"""
测试process_single_obj.py脚本的功能
"""
import os
import numpy as np
import subprocess


def run_command(command):
    """运行命令并打印输出"""
    print(f"运行命令: {' '.join(command)}")
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    if result.stdout:
        print(f"输出:\n{result.stdout}")
    if result.stderr:
        print(f"错误:\n{result.stderr}")
    
    return result.returncode


def test_command_line():
    """测试命令行方式调用"""
    print("\n=== 测试命令行方式调用 ===")
    
    # 假设我们有一个示例OBJ文件路径
    # 您可以修改此路径为您实际的OBJ文件路径
    obj_file = "/home/ktj/Projects/MeshMamba/dataset/non_texture/Apple_v01_l3.obj"  # 请修改为实际路径
    output_file = "./test_output.npz"
    
    if not os.path.exists(obj_file):
        print(f"警告: 示例OBJ文件不存在: {obj_file}")
        print("请修改脚本中的obj_file变量为您实际的OBJ文件路径")
        return False
    
    # 运行命令行工具
    command = [
        "python", "process_single_obj.py",
        "--input", obj_file,
        "--output", output_file,
        "--device", "cpu"
    ]
    
    return_code = run_command(command)
    
    if return_code == 0 and os.path.exists(output_file):
        print(f"\n命令行测试成功！输出文件已创建: {output_file}")
        
        # 检查输出文件内容
        check_output_file(output_file)
        return True
    else:
        print("命令行测试失败！")
        return False


def check_output_file(file_path):
    """检查输出NPZ文件的内容"""
    try:
        data = np.load(file_path)
        print(f"\n输出文件内容检查:")
        print(f"文件包含的键: {list(data.keys())}")
        
        # 打印一些基本信息
        if 'faces' in data:
            print(f"面数: {data['faces'].shape[0]}")
        if 'verts' in data:
            print(f"顶点数: {data['verts'].shape[0]}")
        if 'ring_1' in data:
            print(f"1-Ring形状: {data['ring_1'].shape}")
        if 'ring_2' in data:
            print(f"2-Ring形状: {data['ring_2'].shape}")
        if 'ring_3' in data:
            print(f"3-Ring形状: {data['ring_3'].shape}")
        
        # 打印一些示例数据
        if 'ring_1' in data and data['ring_1'].shape[0] > 0:
            print(f"前3个面的1-Ring邻居示例:")
            for i in range(min(3, data['ring_1'].shape[0])):
                print(f"面{i}: {data['ring_1'][i]}")
                
    except Exception as e:
        print(f"检查输出文件时出错: {e}")


def test_api_call():
    """测试API方式调用"""
    print("\n=== 测试API方式调用 ===")
    
    try:
        # 导入函数
        from process_single_obj import process_single_obj
        
        # 假设我们有一个示例OBJ文件路径
        obj_file = "/home/ktj/Projects/MeshMamba/dataset/non_texture/Apple_v01_l3.obj"  # 请修改为实际路径
        output_file = "./test_output_api.npz"
        
        if not os.path.exists(obj_file):
            print(f"警告: 示例OBJ文件不存在: {obj_file}")
            print("请修改脚本中的obj_file变量为您实际的OBJ文件路径")
            return False
        
        # 调用API
        success = process_single_obj(
            input_path=obj_file,
            output_path=output_file,
            device='cpu',
            normalize=True
        )
        
        if success and os.path.exists(output_file):
            print(f"API调用测试成功！输出文件已创建: {output_file}")
            
            # 检查输出文件内容
            check_output_file(output_file)
            return True
        else:
            print("API调用测试失败！")
            return False
            
    except ImportError as e:
        print(f"导入process_single_obj模块时出错: {e}")
        return False
    except Exception as e:
        print(f"API调用测试时出错: {e}")
        return False


def create_example_usage():
    """创建示例用法文档"""
    print("\n=== 示例用法 ===")
    
    example = """
# 示例1: 命令行方式处理单个OBJ文件
python process_single_obj.py --input path/to/your/model.obj --output path/to/save/output.npz

# 示例2: 指定设备和是否归一化
python process_single_obj.py --input model.obj --output result.npz --device cpu --no-normalize

# 示例3: 在Python代码中调用API
from process_single_obj import process_single_obj

# 处理OBJ文件
success = process_single_obj(
    input_path="path/to/your/model.obj",  # 输入OBJ文件路径
    output_path="path/to/save/output.npz",  # 输出NPZ文件路径（可选）
    device="cpu",  # 计算设备
    normalize=True  # 是否归一化
)

if success:
    print("处理成功！")
else:
    print("处理失败！")

# 示例4: 加载处理后的结果
data = np.load("path/to/save/output.npz")
faces = data['faces']  # 面数据
verts = data['verts']  # 顶点数据
ring_1 = data['ring_1']  # 1-Ring邻居
ring_2 = data['ring_2']  # 2-Ring邻居
ring_3 = data['ring_3']  # 3-Ring邻居
neighbors = data['neighbors']  # 邻居数据
    """
    
    print(example)


def main():
    """主函数"""
    print("开始测试process_single_obj.py...")
    
    # 创建一个简单的示例.obj文件（如果没有的话）
    create_sample_obj()
    
    # 测试命令行方式
    # command_line_success = test_command_line()
    
    # 测试API方式
    # api_success = test_api_call()
    
    # 展示示例用法
    create_example_usage()
    
    print("\n测试完成！请根据上面的示例使用process_single_obj.py处理您的OBJ文件。")


def create_sample_obj():
    """创建一个简单的示例.obj文件用于测试"""
    sample_path = "./sample_cube.obj"
    
    if not os.path.exists(sample_path):
        print(f"创建示例OBJ文件: {sample_path}")
        
        # 一个简单的立方体OBJ文件内容
        cube_obj = """
# 简单立方体OBJ文件
v 1.0 1.0 1.0
v 1.0 1.0 -1.0
v 1.0 -1.0 1.0
v 1.0 -1.0 -1.0
v -1.0 1.0 1.0
v -1.0 1.0 -1.0
v -1.0 -1.0 1.0
v -1.0 -1.0 -1.0
f 1 2 4 3
f 5 6 8 7
f 1 2 6 5
f 3 4 8 7
f 1 3 7 5
f 2 4 8 6
"""
        
        with open(sample_path, 'w') as f:
            f.write(cube_obj)
        
        print(f"示例立方体OBJ文件已创建，请运行以下命令进行测试：")
        print(f"python process_single_obj.py --input {sample_path} --output sample_output.npz")


if __name__ == "__main__":
    main()