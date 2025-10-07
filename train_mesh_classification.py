import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from data import MeshClassificationDataset

# 简单的网格分类模型示例
class SimpleMeshClassifier(nn.Module):
    def __init__(self, num_classes=40, input_dim=15):  # 3(中心) + 3(法向) + 9(角) = 15
        super(SimpleMeshClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, centers, normals, corners):
        # 将所有特征连接在一起
        x = torch.cat([centers, normals, corners], dim=2)  # (B, F, 15)
        # 全局平均池化所有面
        x = torch.mean(x, dim=1)  # (B, 15)
        
        # 前向传播
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

def train_model():
    # 配置参数
    cfg = {
        'data_root': '/mnt/newdisk/ktj/Mesh/dataset_preprocessed/Manifold40',
        'max_faces': 500,
        'augment_data': True,
        'augment_vert': False,
        'augment_rotation': True,
        'jitter_sigma': 0.01,
        'jitter_clip': 0.05,
        'batch_size': 4,
        'num_workers': 0,  # 设置为0以避免多进程问题
        'learning_rate': 0.001,
        'num_epochs': 10
    }
    
    # 创建数据集
    print("Loading datasets...")
    train_dataset = MeshClassificationDataset(cfg, part='train')
    
    # 创建验证数据集
    val_dataset = MeshClassificationDataset(cfg, part='test')
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=True, 
        num_workers=cfg['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg['batch_size'], 
        shuffle=False, 
        num_workers=cfg['num_workers'],
        pin_memory=True
    )
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    print(f"Categories: {train_dataset.get_categories()}")
    
    # 创建模型
    model = SimpleMeshClassifier(num_classes=train_dataset.get_num_classes())
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    
    # 训练循环
    print("Starting training...")
    for epoch in range(cfg['num_epochs']):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, batch in enumerate(train_loader):
            # 获取数据
            centers = batch['centers']  # (B, F, 3)
            normals = batch['normals']  # (B, F, 3)
            corners = batch['corners']  # (B, F, 9)
            labels = batch['label']      # (B,)
            
            # 前向传播
            outputs = model(centers, normals, corners)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 打印批次信息
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{cfg["num_epochs"]}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {running_loss/10:.4f}, Accuracy: {100*correct/total:.2f}%')
                running_loss = 0.0
        
        # 每个epoch结束后打印训练信息
        train_acc = 100 * correct / total
        print(f'End of Epoch [{epoch+1}/{cfg["num_epochs"]}], Training Accuracy: {train_acc:.2f}%')
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                # 获取数据
                centers = batch['centers']  # (B, F, 3)
                normals = batch['normals']  # (B, F, 3)
                corners = batch['corners']  # (B, F, 9)
                labels = batch['label']      # (B,)
                
                # 前向传播
                outputs = model(centers, normals, corners)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        print(f'End of Epoch [{epoch+1}/{cfg["num_epochs"]}], Validation Accuracy: {val_acc:.2f}%')
    
    print("Training completed!")
    
    # 保存模型
    torch.save(model.state_dict(), 'mesh_classifier.pth')
    print("Model saved as mesh_classifier.pth")

if __name__ == "__main__":
    train_model()