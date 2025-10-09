import copy
import os
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import math
import numpy as np
from data.MeshClassificationDataset import MeshClassificationDataset
# from models import *
# from utils.retrival import append_feature, calculate_map
from utils.loss_function import *
import torch.nn.init as init
import yaml
import datetime
import sys
# import time
import copy
from models.MambaMesh import MambaMesh
# from models.MambaMesh_Modified import MambaMesh
from tqdm import tqdm

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal


def train_model(model, criterion, optimizer, scheduler, cfg, data_loader, data_set, visual_path, exp_id):
    best_acc = 0.0  
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # 为了混淆矩阵统计
    all_preds = []
    all_labels = []

    for epoch in range(1, cfg['max_epoch'] + 1):

        print('-' * 60)
        print('Epoch: {} / {}'.format(epoch, cfg['max_epoch']))
        print('-' * 60)

        for phrase in ['train', 'test']:
            
            print('Phrase: {}'.format(phrase))
            print('-' * 60)
            current_preds = []
            current_labels = []
            
            if phrase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            # 使用tqdm添加进度条
            for collated_dict in tqdm(data_loader[phrase], desc=f'{phrase} Epoch {epoch}', unit='batch'):
                # 获取数据
                faces = collated_dict['faces'].cuda()
                verts = collated_dict['verts'].cuda()
                centers = collated_dict['centers'].permute(0, 2, 1).cuda()
                normals = collated_dict['normals'].permute(0, 2, 1).cuda()
                corners = collated_dict['corners'].permute(0, 2, 1).cuda()
                targets = collated_dict['label'].cuda()
                
                # Manifold40数据集没有这些特征，设为None或默认值
                neighbor_index = torch.zeros_like(faces) if 'neighbors' not in collated_dict else collated_dict['neighbors'].cuda()
                ring_1 = torch.zeros_like(faces) if 'ring_1' not in collated_dict else collated_dict['ring_1'].cuda()
                ring_2 = torch.zeros_like(faces) if 'ring_2' not in collated_dict else collated_dict['ring_2'].cuda()
                ring_3 = torch.zeros_like(faces) if 'ring_3' not in collated_dict else collated_dict['ring_3'].cuda()
                
                # 纹理相关特征也设为默认值
                # texture = torch.zeros(1, 3, 256, 256).cuda()
                # uv_grid = torch.zeros(1, 1, 1, 2).cuda()

                with torch.set_grad_enabled(phrase == 'train'):
                    # 对于分类任务，我们期望模型输出分类结果
                    outputs = model(verts=verts,
                                    faces=faces,
                                    centers=centers,
                                    normals=normals,
                                    corners=corners,
                                    neighbor_index=neighbor_index,
                                    ring_1=ring_1,
                                    ring_2=ring_2,
                                    ring_3=ring_3,
                                    )
                    
                    # 如果模型输出是元组，取第一个元素作为分类结果
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # 计算分类损失
                    loss = criterion(outputs, targets)

                    if phrase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # 统计准确率
                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * targets.size(0)
                    running_corrects += torch.sum(preds == targets.data)
                    
                    # 保存预测和标签用于后续分析
                    current_preds.extend(preds.cpu().numpy())
                    current_labels.extend(targets.cpu().numpy())

            # 计算当前epoch的损失和准确率
            epoch_loss = running_loss / len(data_set[phrase])
            epoch_acc = running_corrects.double() / len(data_set[phrase])
            
            # 保存所有预测和标签
            all_preds.extend(current_preds)
            all_labels.extend(current_labels)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phrase, epoch_loss, epoch_acc))

            # 在训练集上更新学习率
            if phrase == 'train':
                scheduler.step()

            # 在测试集上保存最佳模型
            if phrase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best test accuracy improved to: {:.4f}'.format(best_acc))

            # 按配置保存模型检查点
            if phrase == 'test' and epoch % cfg.get('save_steps', 10) == 0:
                torch.save(copy.deepcopy(model.state_dict()),
                           os.path.join(cfg['ckpt_root'], 'epoch_{}.pkl'.format(epoch)))
    



    # 保存最终的混淆矩阵数据（可选）
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)
    
    np.savez(os.path.join(visual_path, 'predictions_exp_{}.npz'.format(exp_id)), 
             preds=all_preds, labels=all_labels)

    print('Exp {} Best test accuracy: {:.4f}'.format(exp_id, best_acc))
    print('Config: {}'.format(cfg))

    return best_model_wts

    print('Exp {} Best val mse: {:.4f}'.format(exp_id, best_acc))
    print('Config: {}'.format(cfg))

    return best_model_wts


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            # 使用Xavier初始化权重
            init.xavier_uniform_(m.weight)
            # 初始化偏置为零
            if m.bias is not None:
                init.constant_(m.bias, 0)



if __name__ == '__main__':
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # 设置随机种子以保证结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 加载配置文件
    with open("config/Manifold40.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
    
    # 设置分类任务的损失函数
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['training'].get('label_smoothing', 0.1))

    # 创建日志目录
    script_name = os.path.basename(__file__)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 使用下划线代替空格
    file_path = "log/" + str('Manifold40_Classification') + "_" + str(current_time)
    os.makedirs(file_path, exist_ok=True)

    # 分类任务通常不需要多次重复实验，可以根据需要调整
    num_experiments = cfg['training'].get('num_experiments', 1)
    for exp_id in range(num_experiments):
        current_log = Logger(filename="{}/{}.log".format(file_path, exp_id))
        sys.stdout = current_log
        print(script_name)
        print("Starting experiment {}/{}".format(exp_id+1, num_experiments))
        print("Dataset root: {}".format(cfg['dataset']['data_root']))

        # 创建Manifold40数据集实例
        data_set = {
            x: MeshClassificationDataset(cfg=cfg['dataset'], part=x) for x in ['train', 'test']
        }
        
        # 打印数据集信息
        print("Number of training samples: {}".format(len(data_set['train'])))
        print("Number of test samples: {}".format(len(data_set['test'])))
        print("Number of classes: {}".format(len(data_set['train'].categories)))
        # print("Class names: {}".format(data_set['train'].get_class_names()))

        # 创建数据加载器
        data_loader = {
            x: data.DataLoader(
                data_set[x], 
                batch_size=cfg['training'].get('batch_size', 32), 
                num_workers=cfg['training'].get('num_workers', 4), 
                shuffle=(x == 'train'), 
                pin_memory=cfg['training'].get('pin_memory', True)
            ) for x in ['train', 'test']
        }

        # 创建可视化路径
        visual_path = "{}/visualization_{}/".format(file_path, exp_id)
        os.makedirs(visual_path, exist_ok=True)

        # 准备模型
        # 确保模型配置中分类维度正确设置
        if 'cls_dim' not in cfg['mamba'] or cfg['mamba']['cls_dim'] != data_set['train'].num_classes:
            cfg['mamba']['cls_dim'] = data_set['train'].num_classes
            print("Setting model cls_dim to number of classes: {}".format(cfg['mamba']['cls_dim']))
        
        
        
        model = MambaMesh(cfg['mamba']).cuda()
        
        # 打印模型参数量
        total = sum([param.nelement() for param in model.parameters()])
        print("Number of parameters: %.2fM" % (total / 1e6))
        
        # 初始化权重
        initialize_weights(model)
        
        # 如果有多个GPU，可以使用数据并行
        # if torch.cuda.device_count() > 1:
        #     print("Using {} GPUs for training".format(torch.cuda.device_count()))
        #     model = nn.DataParallel(model)

        # 创建优化器
        optimizer_type = cfg['training'].get('optimizer', 'adam')
        lr = cfg['training'].get('lr', 0.001)
        weight_decay = cfg['training'].get('weight_decay', 0.0001)
        
        if optimizer_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), 
                lr=lr, 
                momentum=cfg['training'].get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adam':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )

        # 创建学习率调度器
        scheduler_type = cfg['training'].get('scheduler', 'cos')
        max_epoch = cfg['training'].get('max_epoch', 100)
        
        if scheduler_type == 'multistep':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer, 
                milestones=cfg['training'].get('milestones', [50, 80])
            )
        elif scheduler_type == 'cos':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=max_epoch
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=cfg['training'].get('step_size', 50), 
                gamma=cfg['training'].get('gamma', 0.1)
            )

        # 创建检查点目录
        ckpt_root = cfg['checkpoint'].get('ckpt_root', 'ckpt_root/manifold40_classification')
        os.makedirs(ckpt_root, exist_ok=True)
        
        # 开始训练
        best_model_wts = train_model(
            model=model, 
            criterion=criterion, 
            optimizer=optimizer, 
            scheduler=scheduler, 
            cfg=cfg, 
            data_loader=data_loader,
            data_set=data_set,
            visual_path=visual_path,
            exp_id=exp_id
        )
        
        # 保存最佳模型
        torch.save(best_model_wts, os.path.join(ckpt_root, 'manifold40_best_exp{}.pkl'.format(exp_id)))
        print("Model saved to {}".format(os.path.join(ckpt_root, 'manifold40_best_exp{}.pkl'.format(exp_id))))
        
        current_log.close()

    print("All experiments completed!")
