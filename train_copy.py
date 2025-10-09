# train_cls_dp_resume_metrics.py
"""
DataParallel multi-GPU training script with:
 - automatic resume
 - checkpointing (model + optimizer + scheduler + epoch + best_acc)
 - logging + TensorBoard
 - periodic checkpoint saving
 - global evaluation metrics: precision/recall/f1 and confusion matrix saved as image
"""
# CUDA_VISIBLE_DEVICES=0,1
import copy
import os
import random
import sys
import time
import datetime
import yaml
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.MeshClassificationDataset_copy import MeshClassificationDataset
from models.MambaMesh import MambaMesh
from utils.loss_function import *
import multiprocessing as mp
from utils.file_utils import fpath, get_dataset_paths
# ---------------- Logger ----------------
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

# ---------------- Helpers ----------------
def save_checkpoint(state, ckpt_dir, filename="checkpoint.pth"):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, filename)
    torch.save(state, path)
    return path

def find_latest_ckpt(ckpt_root):
    files = glob.glob(os.path.join(ckpt_root, "*.pth"))
    if not files:
        return None
    return sorted(files, key=os.path.getmtime)[-1]

def plot_and_save_confusion_matrix(y_true, y_pred, classes, out_path, normalize=True, title="Confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        cm = cm.astype('float') / np.where(cm_sum == 0, 1, cm_sum)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', vmin=0.0, vmax=1.0)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=90, fontsize=6)
    ax.set_yticklabels(classes, fontsize=6)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=6)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

# ---------------- Main Training ----------------
def train_model(cfg, exp_id):    
    # Define device based on available GPUs
    if torch.cuda.is_available():
        device = torch.device(f"cuda:1")
    else:
        device = torch.device("cpu")
        use_dp = False
        print("No GPU detected. Using CPU training.")

    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # dataset path 
    data_root = cfg['dataset']['data_root']
    dataset_paths = get_dataset_paths(data_root)


    # datasets & loaders
    data_set = {x: MeshClassificationDataset(cfg=cfg['dataset'], part=x, mesh_paths=dataset_paths[x]) for x in ['train', 'test']}
    data_loader = {
        'train': data.DataLoader(
            data_set['train'],
            batch_size=cfg['training'].get('batch_size', 128),
            shuffle=True,
            num_workers=cfg['training'].get('num_workers', 4),
            pin_memory=cfg['training'].get('pin_memory', True),
            # multiprocessing_context='spawn'
        ),
        'test': data.DataLoader(
            data_set['test'],
            batch_size=cfg['training'].get('batch_size', 128),
            shuffle=False,
            num_workers=cfg['training'].get('num_workers', 4),
            pin_memory=cfg['training'].get('pin_memory', True)
        )
    }

    # model
    cfg['mamba']['cls_dim'] = data_set['train'].num_classes
    model = MambaMesh(cfg['mamba']).to(device)
    # if use_dp:
    #     model = DataParallel(model, device_ids)
    # print(f"Using {torch.cuda.device_count()} GPU(s)")

    # optimizer/scheduler/criterion
    optimizer_type = cfg['training'].get('optimizer', 'adam')
    lr = cfg['training'].get('lr', 0.001)
    weight_decay = cfg['training'].get('weight_decay', 1e-4)
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=cfg['training'].get('momentum', 0.9), weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    max_epoch = cfg['training'].get('max_epoch', 100)
    scheduler_type = cfg['training'].get('scheduler', 'cos')
    if scheduler_type == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg['training'].get('milestones', [50, 80]))
    elif scheduler_type == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg['training'].get('step_size', 50),
                                              gamma=cfg['training'].get('gamma', 0.1))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg['training'].get('label_smoothing', 0.1)).to(device)

    # logging
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_root = cfg.get('log_root', 'log')
    log_dir = os.path.join(log_root, f"Manifold40_Classification_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    logger = Logger(os.path.join(log_dir, f"exp_{exp_id}.log"))
    sys.stdout = logger
    tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))

    print(f"Start experiment {exp_id} | GPUs: {torch.cuda.device_count()} | Logdir: {log_dir}")
    print(f"Train samples: {len(data_set['train'])}, Test samples: {len(data_set['test'])}, Classes: {data_set['train'].num_classes}")

    # checkpoints / resume
    ckpt_root = cfg['checkpoint'].get('ckpt_root', 'ckpt_root/manifold40_classification')
    os.makedirs(ckpt_root, exist_ok=True)
    resume_flag = cfg['checkpoint'].get('resume', False)
    resume_path = cfg['checkpoint'].get('resume_path', None)
    start_epoch = 1
    best_acc = 0.0

    if resume_flag and (resume_path is None or not os.path.exists(resume_path)):
        latest = find_latest_ckpt(ckpt_root)
        if latest is not None:
            resume_path = latest
            print(f"[Auto-resume] Found latest checkpoint: {resume_path}")

    if resume_flag and resume_path is not None and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        try:
            model.load_state_dict(ckpt['model_state'])
        except Exception as e:
            model.load_state_dict(ckpt['model_state'], strict=False)
            print("Warning: loaded checkpoint with strict=False due to mismatch:", e)
        if 'optimizer_state' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state'])
            except Exception as e:
                print("Warning: failed to load optimizer state:", e)
        if 'scheduler_state' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state'])
            except Exception as e:
                print("Warning: failed to load scheduler state:", e)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"Resume from {resume_path} -> starting epoch {start_epoch}, best_acc={best_acc}")

    best_model_wts = copy.deepcopy(model.state_dict())
    total_train_samples = len(data_set['train'])
    total_test_samples = len(data_set['test'])
    save_every = cfg['checkpoint'].get('save_steps', 5)
    class_names = getattr(data_set['train'], 'categories', None) or [str(i) for i in range(cfg['mamba']['cls_dim'])]

    # training loop
    for epoch in range(start_epoch, max_epoch + 1):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        loop = tqdm(data_loader['train'], desc=f"Train Epoch {epoch}")
        for collated_dict in loop:
            faces = collated_dict['faces'].to(device)
            verts = collated_dict['verts'].to(device)
            centers = collated_dict['centers'].permute(0, 2, 1).to(device)
            normals = collated_dict['normals'].permute(0, 2, 1).to(device)
            corners = collated_dict['corners'].permute(0, 2, 1).to(device)
            targets = collated_dict['label'].to(device)

            neighbor_index = collated_dict.get('neighbors', torch.zeros_like(faces)).to(device)
            ring_1 = collated_dict.get('ring_1', torch.zeros_like(faces)).to(device)
            ring_2 = collated_dict.get('ring_2', torch.zeros_like(faces)).to(device)
            ring_3 = collated_dict.get('ring_3', torch.zeros_like(faces)).to(device)

            outputs = model(verts=verts, faces=faces, centers=centers,
                            normals=normals, corners=corners,
                            neighbor_index=neighbor_index,
                            ring_1=ring_1, ring_2=ring_2, ring_3=ring_3)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * targets.size(0)
            running_corrects += torch.sum(preds == targets.data).item()

        scheduler.step()
        epoch_loss = running_loss / total_train_samples
        epoch_acc = running_corrects / total_train_samples
        print(f"[Epoch {epoch}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        tb_writer.add_scalar("Train/Loss", epoch_loss, epoch)
        tb_writer.add_scalar("Train/Acc", epoch_acc, epoch)

        # evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            loop = tqdm(data_loader['test'], desc=f"Test Epoch {epoch}")
            for collated_dict in loop:
                faces = collated_dict['faces'].to(device)
                verts = collated_dict['verts'].to(device)
                centers = collated_dict['centers'].permute(0, 2, 1).to(device)
                normals = collated_dict['normals'].permute(0, 2, 1).to(device)
                corners = collated_dict['corners'].permute(0, 2, 1).to(device)
                targets = collated_dict['label'].to(device)

                neighbor_index = collated_dict.get('neighbors', torch.zeros_like(faces)).to(device)
                ring_1 = collated_dict.get('ring_1', torch.zeros_like(faces)).to(device)
                ring_2 = collated_dict.get('ring_2', torch.zeros_like(faces)).to(device)
                ring_3 = collated_dict.get('ring_3', torch.zeros_like(faces)).to(device)

                outputs = model(verts=verts, faces=faces, centers=centers,
                                normals=normals, corners=corners,
                                neighbor_index=neighbor_index,
                                ring_1=ring_1, ring_2=ring_2, ring_3=ring_3)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(targets.cpu().numpy().tolist())

        # metrics
        report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        acc = np.mean(np.array(all_preds) == np.array(all_labels))

        print(f"[Epoch {epoch}] Test Acc: {acc:.4f} Precision: {precision:.4f} Recall: {recall:.4f} F1: {f1:.4f}")
        tb_writer.add_scalar("Test/Acc", acc, epoch)
        tb_writer.add_scalar("Test/Precision_weighted", precision, epoch)
        tb_writer.add_scalar("Test/Recall_weighted", recall, epoch)
        tb_writer.add_scalar("Test/F1_weighted", f1, epoch)

        # save classification report & confusion matrix
        report_txt = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
        with open(os.path.join(log_dir, f"classification_report_epoch{epoch}.txt"), "w") as f:
            f.write(report_txt)
        cm_path = os.path.join(log_dir, f"confusion_matrix_epoch{epoch}.png")
        plot_and_save_confusion_matrix(all_labels, all_preds, class_names, cm_path, normalize=True,
                                       title=f"Confusion matrix (epoch {epoch})")
        print(f"Saved confusion matrix -> {cm_path}")

        # best model
        if acc > best_acc:
            best_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            print(f"New best acc: {best_acc:.4f}")

        # periodic checkpoint
        if epoch % save_every == 0 or epoch == max_epoch:
            ckpt_name = f"exp{exp_id}_epoch{epoch}.pth"
            ckpt_state = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc': best_acc
            }
            save_path = save_checkpoint(ckpt_state, ckpt_root, ckpt_name)
            print(f"Saved checkpoint: {save_path}")

    # final save of best model
    final_name = f"exp{exp_id}_best.pth"
    final_state = {
        'epoch': epoch,
        'model_state': best_model_wts,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_acc': best_acc
    }
    final_path = save_checkpoint(final_state, ckpt_root, final_name)
    print(f"Saved best model to {final_path}")
    tb_writer.close()
    logger.close()

# ---------------- Startup ----------------
def main():
    with open("config/Manifold40.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.loader.SafeLoader)
    train_model(cfg, exp_id=0)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
