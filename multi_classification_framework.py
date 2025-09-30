#!/usr/bin/env python3
"""
multi_classification_framework.py

一键式多分类训练框架（单文件）：
- 支持 timm 中大量模型（通过 --model 指定）
- 使用文件夹组织的数据集（ImageFolder 风格）：root/class_x/xxx.jpg
- 自动数据集划分（train/val/test 按比例划分）
- 训练 / 验证 / 测试循环（支持混合精度、多GPU）
- 保存最佳 checkpoint
- 记录训练日志（CSV + TensorBoard 可选）
- 导出训练/验证/测试的指标变化图（PNG）

依赖：
  torch, torchvision, timm, sklearn, pandas, matplotlib, tqdm
可选：tensorboard

示例：
  python multi_classification_framework.py \
    --data_dir ./data \
    --model resnet50 \
    --epochs 30 \
    --batch_size 32 \
    --output_dir ./output

"""

import os
import argparse
import shutil
from pathlib import Path
from datetime import datetime
import random
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

try:
    import timm
except Exception:
    timm = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# ----------------------------- Dataset helper -----------------------------
class ImagePathDataset(Dataset):
    """Dataset that loads images from a list of (path, label) pairs and applies transforms."""
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


def scan_imagefolder(root):
    """Scan a folder with structure root/class_x/xxx.jpg and return list of (path, label) and class names."""
    root = Path(root)
    classes = [d.name for d in sorted(root.iterdir()) if d.is_dir()]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    items = []
    for c in classes:
        for p in (root / c).rglob('*'):
            if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                items.append((str(p), class_to_idx[c]))
    return items, classes


# ----------------------------- Utils -----------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------- Training / Eval -----------------------------

def train_one_epoch(model, criterion, optimizer, dataloader, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(dataloader, desc='Train', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=running_loss/total, acc=correct/total)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    pbar = tqdm(dataloader, desc='Eval ', leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    if total == 0:
        return 0, 0, np.array([]), np.array([])
    return running_loss / total, correct / total, np.concatenate(all_preds), np.concatenate(all_labels)


# ----------------------------- Plotting -----------------------------

def plot_metrics(history, out_dir):
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    # loss
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    plt.plot(df['epoch'], df['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'), bbox_inches='tight')
    plt.close()
    # acc
    plt.figure()
    plt.plot(df['epoch'], df['train_acc'], label='train_acc')
    plt.plot(df['epoch'], df['val_acc'], label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'acc_curve.png'), bbox_inches='tight')
    plt.close()


# ----------------------------- Main pipeline -----------------------------

def build_model(model_name, num_classes, pretrained=True):
    if timm is None:
        raise RuntimeError('timm is required. Please install: pip install timm')
    # ------------------------------------------
    # 模型接入位置：可以在这里添加自定义模型
    # 示例：
    # if model_name == 'your_custom_model':
    #     model = YourCustomModel(num_classes=num_classes)
    #     return model
    # ------------------------------------------
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def make_transforms(image_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])


def split_items(items, val_ratio=0.1, test_ratio=0.1, stratify=True, seed=42):
    # items: list of (path, label)
    paths = [p for p, l in items]
    labels = [l for p, l in items]
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        paths, labels, test_size=test_ratio, random_state=seed, stratify=labels if stratify else None)
    relative_val = val_ratio / (1 - test_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, test_size=relative_val, random_state=seed, stratify=train_val_labels if stratify else None)
    train_items = list(zip(train_paths, train_labels))
    val_items = list(zip(val_paths, val_labels))
    test_items = list(zip(test_paths, test_labels))
    return train_items, val_items, test_items




def main(args):
    # args = parse_args()
    set_seed(args.seed)

    # prepare output dir
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(args.output_dir, f"run_{args.model}_{now}")
    os.makedirs(out_dir, exist_ok=True)

    # scan dataset
    items, classes = scan_imagefolder(args.data_dir)
    if len(items) == 0:
        raise RuntimeError('No images found in data_dir')
    num_classes = len(classes)
    save_json({'classes': classes}, os.path.join(out_dir, 'classes.json'))

    # split
    train_items, val_items, test_items = split_items(items, val_ratio=args.val_ratio, test_ratio=args.test_ratio, stratify=True, seed=args.seed)
    print(f"Found {len(items)} images, {num_classes} classes -> train {len(train_items)}, val {len(val_items)}, test {len(test_items)}")

    # transforms
    train_transform = make_transforms(args.image_size, train=True)
    val_transform = make_transforms(args.image_size, train=False)

    # datasets and loaders
    train_ds = ImagePathDataset(train_items, transform=train_transform)
    val_ds = ImagePathDataset(val_items, transform=val_transform)
    test_ds = ImagePathDataset(test_items, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    device = args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu'
    model = build_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)
    if torch.cuda.device_count() > 1 and device != 'cpu':
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device != 'cpu')

    # logging
    writer = None
    if SummaryWriter is not None and not args.no_tb:
        writer = SummaryWriter(log_dir=os.path.join(out_dir, 'tensorboard'))

    history = []
    best_val_acc = 0.0
    start_epoch = 1

    # resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt.get('epoch', 1) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"Resumed from {args.resume}, start_epoch={start_epoch}, best_val_acc={best_val_acc}")

    # training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, criterion, optimizer, train_loader, device, scaler if args.amp else None)
        val_loss, val_acc, _, _ = evaluate(model, criterion, val_loader, device)

        # scheduler step
        scheduler.step(val_acc)

        print(f"Epoch {epoch}/{args.epochs}  Train loss: {train_loss:.4f} acc: {train_acc:.4f}  Val loss: {val_loss:.4f} acc: {val_acc:.4f}")

        # logging
        if writer:
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('acc/train', train_acc, epoch)
            writer.add_scalar('acc/val', val_acc, epoch)

        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})

        # save checkpoint
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
        }
        ckpt_path = os.path.join(out_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(ckpt, ckpt_path)

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(out_dir, 'best.pth')
            torch.save(ckpt, best_path)
            print(f"New best val acc: {best_val_acc:.4f}, saved to {best_path}")

        # periodic save smaller
        if epoch % args.save_freq == 0 and epoch > 0:
            small_ckpt = {k: v for k, v in ckpt.items() if k != 'optimizer_state'}
            torch.save(small_ckpt, os.path.join(out_dir, f'ckpt_epoch_{epoch}_small.pth'))

    # finish
    if writer:
        writer.close()

    # plot metrics
    plot_metrics(history, out_dir)

    # final test evaluation with best checkpoint
    best_ckpt = torch.load(os.path.join(out_dir, 'best.pth'), map_location=device)
    model_state = best_ckpt['model_state']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

    test_loss, test_acc, preds, labels = evaluate(model, criterion, test_loader, device)
    print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")
    with open(os.path.join(out_dir, 'test_results.txt'), 'w') as f:
        f.write(f"test_loss: {test_loss}\ntest_acc: {test_acc}\n")

    # save confusion matrix csv
    try:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(labels, preds)
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(out_dir, 'confusion_matrix.csv'))
    except Exception:
        pass

    print(f"All artifacts saved to {out_dir}")



if __name__ == '__main__':
    # 在此处配置运行参数
    class Args:
        pass
    
    args = Args()
    # 数据相关参数
    args.data_dir = './classificationDataset'  # 数据集路径，应包含子文件夹表示各类别
    args.model = 'resnet18'   # 模型名称
    args.image_size = 224     # 输入图像尺寸
    args.pretrained = False    # 是否使用预训练权重
    args.val_ratio = 0.1      # 验证集比例
    args.test_ratio = 0.1     # 测试集比例
    
    # 训练相关参数
    args.epochs = 20          # 训练轮数
    args.batch_size = 32      # 批次大小
    args.lr = 1e-3            # 学习率
    args.weight_decay = 1e-4  # 权重衰减
    args.seed = 42            # 随机种子
    
    # 硬件相关参数
    args.output_dir = './output'  # 输出目录
    args.num_workers = 4          # 数据加载器的工作进程数
    args.device = 'cuda'          # 设备类型 ('cuda' 或 'cpu')
    args.amp = False              # 是否使用混合精度训练
    
    # 其他参数
    args.resume = None        # 恢复训练的检查点路径
    args.save_freq = 5        # 定期保存检查点的频率
    args.no_tb = False        # 是否禁用 TensorBoard 日志
    
    # ------------------------------------------
    # 模型接入位置：可以在上面添加自定义模型参数
    # 并在 build_model 函数中添加对应的模型构建逻辑
    # ------------------------------------------
    
    main(args)
