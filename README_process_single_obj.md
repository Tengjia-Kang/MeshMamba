# 从OBJ文件开始处理网格数据

这个工具允许您从OBJ文件开始处理，计算网格的neighbors、ring_1、ring_2、ring_3等数据，适用于无纹理无颜色的mesh。

## 功能特点

- 支持从单个OBJ文件开始处理
- 提供命令行界面和Python API两种使用方式
- 可以选择是否对网格进行归一化处理
- 支持CPU和GPU(CUDA)计算
- 输出包含faces、verts、neighbors、ring_1、ring_2、ring_3等数据

## 安装依赖

确保您已安装所需的依赖库：

```bash
pip install numpy torch torch-geometric trimesh pytorch3d
```

## 使用方法

### 方法1: 命令行方式

基本用法：

```bash
python process_single_obj.py --input path/to/your/model.obj --output path/to/save/output.npz
```

完整选项：

```bash
python process_single_obj.py \
  --input path/to/your/model.obj \
  --output path/to/save/output.npz \
  --device cpu \
  --no-normalize
```

参数说明：
- `--input`, `-i`: 输入OBJ文件的路径（必需）
- `--output`, `-o`: 输出NPZ文件的路径（可选，默认在输入文件同目录下）
- `--device`: 计算设备，可选'cpu'或'cuda'（默认'cpu'）
- `--no-normalize`: 禁用网格归一化（默认会进行归一化）

### 方法2: Python API方式

在您的Python代码中导入并使用：

```python
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
```

## 输出文件说明

处理完成后，输出的NPZ文件包含以下数据：

- `faces`: 网格的面数据，形状为 [面数, 3]
- `verts`: 网格的顶点数据，形状为 [顶点数, 3]
- `neighbors`: 每个面的直接邻居，形状为 [面数, 3]
- `ring_1`: 1-Ring邻居，形状为 [面数, 3]
- `ring_2`: 2-Ring邻居，形状为 [面数, 6]
- `ring_3`: 3-Ring邻居，形状为 [面数, 12]

## 示例

1. 首先创建一个简单的示例OBJ文件：

```bash
python test_process_single_obj.py
```

2. 处理示例OBJ文件：

```bash
python process_single_obj.py --input sample_cube.obj --output sample_output.npz
```

3. 加载和使用处理结果：

```python
import numpy as np

data = np.load("sample_output.npz")
print("文件包含的键:", list(data.keys()))
print("面数:", data['faces'].shape[0])
print("顶点数:", data['verts'].shape[0])
print("1-Ring形状:", data['ring_1'].shape)
print("前3个面的1-Ring邻居:")
for i in range(min(3, data['ring_1'].shape[0])):
    print(f"面{i}: {data['ring_1'][i]}")
```

## 注意事项

1. 确保输入的OBJ文件格式正确
2. 如果使用CUDA设备，请确保您的环境支持PyTorch CUDA
3. 对于特别大的网格模型，可能需要更多的内存
4. 归一化操作会将模型移动到原点并缩放到单位球内，这有助于后续处理

## 测试脚本

`test_process_single_obj.py` 文件提供了测试功能和使用示例：

```bash
python test_process_single_obj.py
```

此脚本会创建一个示例立方体OBJ文件，并展示如何使用主脚本进行处理。