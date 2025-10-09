import os
import numpy as np

def check_npz_faces(folder_path):
    """
    检查文件夹中所有NPZ文件的faces数组长度是否为500
    若不等于500，则打印该文件所在的文件夹路径
    """
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查是否为npz文件
        if filename.endswith('.npz'):
            file_path = os.path.join(folder_path, filename)
            
            try:
                # 读取npz文件
                with np.load(file_path) as data:
                    if 'ring_1' in data:
                        ring_1 = data['ring_1']
                        # print(f"文件名: {filename}")
                        # print(f"ring_1数量: {ring_1.shape[0]}")
                        # 检查ring_1的长度
                        if ring_1.shape[0] != 500:
                            print(f"文件夹路径: {folder_path}")
                            print(f"  文件名: {filename}")
                            print(f"  ring_1数量: {ring_1.shape[0]}\n")
                    if 'ring_2' in data:
                        ring_2 = data['ring_2']
                        # print(f"文件名: {filename}")
                        # print(f"ring_2数量: {ring_2.shape[0]}")
                        # 检查ring_2的长度
                        if ring_2.shape[0] != 500:
                            print(f"文件夹路径: {folder_path}")
                            print(f"  文件名: {filename}")
                            print(f"  ring_2数量: {ring_2.shape[0]}\n")
                    # 检查是否包含faces键
                    if 'neighbors' in data:
                        neighbors = data['neighbors']
                        # print(f"文件名: {filename}")
                        # print(f"neighbors数量: {neighbors.shape[0]}")
                        # 检查neighbors的长度
                        if neighbors.shape[0] != 500:
                            print(f"文件夹路径: {folder_path}")
                            print(f"  文件名: {filename}")
                            print(f"  neighbors数量: {neighbors.shape[0]}\n")
            except Exception as e:
                print(f"处理文件 {file_path} 时出错: {str(e)}")
        # print(f"检查完成: {folder_path}")
if __name__ == "__main__":
    # 替换为你的npz文件夹路径
    npz_folder = "/home/ktj/Projects/MeshMamba/dataset/processed/Manifold_ringn"
    
    # 检查路径是否存在
    if not os.path.exists(npz_folder):
        print(f"错误: 路径 {npz_folder} 不存在")
    elif not os.path.isdir(npz_folder):
        print(f"错误: {npz_folder} 不是一个文件夹")
    else:
        check_npz_faces(npz_folder)
