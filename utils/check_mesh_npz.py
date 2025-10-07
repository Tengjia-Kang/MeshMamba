import numpy as np

# 加载NPZ文件
data = np.load('/home/ktj/Projects/MeshMamba/text_model.obj.npz')

# 打印文件包含的键
print('文件包含的键:', list(data.keys()))

# 打印基本信息
print('面数:', data['faces'].shape[0])
print('顶点数:', data['verts'].shape[0])
print('1-Ring形状:', data['ring_1'].shape)
print('2-Ring形状:', data['ring_2'].shape)
print('3-Ring形状:', data['ring_3'].shape)
print('neighbors形状:', data['neighbors'].shape)

# 打印前3个面的1-Ring邻居
print('\n前3个面的1-Ring邻居:')
for i in range(min(3, len(data['ring_1']))):
    print(f"面{i}: {data['ring_1'][i]}")

# 打印前3个面的2-Ring邻居
print('\n前3个面的2-Ring邻居:')
for i in range(min(3, len(data['ring_2']))):
    print(f"面{i}: {data['ring_2'][i]}")

# 打印前3个面的3-Ring邻居
print('\n前3个面的3-Ring邻居:')
for i in range(min(3, len(data['ring_3']))):
    print(f"面{i}: {data['ring_3'][i]}")

# 验证数据的一致性
print('\n数据验证:')
# 检查1-Ring每个面是否有3个邻居
if data['ring_1'].shape[1] == 3:
    print('✓ 每个面都有3个1-Ring邻居')
else:
    print('✗ 1-Ring邻居数量不正确')

# 检查2-Ring每个面是否有6个邻居
if data['ring_2'].shape[1] == 6:
    print('✓ 每个面都有6个2-Ring邻居')
else:
    print('✗ 2-Ring邻居数量不正确')

# 检查3-Ring每个面是否有12个邻居
if data['ring_3'].shape[1] == 12:
    print('✓ 每个面都有12个3-Ring邻居')
else:
    print('✗ 3-Ring邻居数量不正确')