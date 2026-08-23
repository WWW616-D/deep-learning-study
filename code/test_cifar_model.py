"""
最简单的 CIFAR-10 模型测试代码
================================
加载 data/cifar_net.pth，在 CIFAR-10 测试集上评估准确率，
并对比「有归一化」和「无归一化」两种评估方式的差异。

用法:
    python test_cifar_model.py            # 从项目根目录 data/ 加载
    python test_cifar_model.py --path D:/py/data/cifar_net.pth
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Net(nn.Module):
    """与 torchtest.py 完全一致的架构"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


def evaluate(model, loader, device):
    """计算整体准确率和每类准确率"""
    model.eval()
    correct = 0
    total = 0
    class_correct = {c: 0 for c in CLASSES}
    class_total = {c: 0 for c in CLASSES}

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for lab, pred in zip(labels, predicted):
                if lab == pred:
                    class_correct[CLASSES[lab]] += 1
                class_total[CLASSES[lab]] += 1

    acc = 100 * correct / total
    print(f"  整体准确率: {correct}/{total} = {acc:.2f}%")
    print("  每类准确率:")
    for c in CLASSES:
        if class_total[c] > 0:
            print(f"    {c:6s}: {100 * class_correct[c] / class_total[c]:.1f}% "
                  f"({class_correct[c]}/{class_total[c]})")
    return acc


def main():
    parser = argparse.ArgumentParser(description='测试 CIFAR-10 模型')
    parser.add_argument('--path', type=str, default='../data/cifar_net.pth',
                        help='模型权重路径 (默认: ../data/cifar_net.pth)')
    parser.add_argument('--data-root', type=str, default='../data',
                        help='CIFAR-10 数据路径 (默认: ../data)')
    parser.add_argument('--samples', type=int, default=1000,
                        help='评估样本数，默认全量 (10000)')
    args = parser.parse_args()

    # 检查模型文件
    if not os.path.exists(args.path):
        print(f"❌ 模型文件不存在: {args.path}")
        # 尝试找候选路径
        for cand in ['cifar_net.pth', '../data/cifar_net.pth',
                     'D:/py/data/cifar_net.pth']:
            if os.path.exists(cand):
                args.path = cand
                print(f"   找到: {cand}")
                break
        else:
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"模型路径: {args.path}")

    # ---- 加载模型 ----
    net = Net()
    state = torch.load(args.path, map_location=device, weights_only=True)
    net.load_state_dict(state)
    net = net.to(device)
    print(f"模型加载成功! 参数量: {sum(p.numel() for p in net.parameters())/1e3:.1f}K\n")

    # ---- 两种评估方式对比 ----
    # 方式 1: 无归一化（gd_uap.py 用的）
    transform_raw = transforms.Compose([transforms.ToTensor()])
    # 方式 2: 与 torchtest.py 训练时一致 (mean=0.5, std=0.5)
    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    for name, transform in [("无归一化 (raw)", transform_raw),
                            ("有归一化 (0.5,0.5)", transform_norm)]:
        try:
            testset = torchvision.datasets.CIFAR10(
                root=args.data_root, train=False, download=False,
                transform=transform)
        except Exception as e:
            print(f"  ⚠️ 无法加载数据: {e}")
            return
        loader = torch.utils.data.DataLoader(
            testset, batch_size=128, shuffle=False, num_workers=0)
        n = min(args.samples, len(testset))
        subset = torch.utils.data.Subset(testset, list(range(n)))
        subloader = torch.utils.data.DataLoader(
            subset, batch_size=128, shuffle=False, num_workers=0)
        print(f"评估方式 [{name}] — 样本数 {n}:")
        evaluate(net, subloader, device)
        print()

    # ---- 权重统计（诊断模型是否真的训练过） ----
    print("\n权重统计诊断:")
    for name_, param in net.named_parameters():
        vals = param.detach().cpu().view(-1)
        print(f"  {name_:15s} shape={str(list(param.shape)):20s} "
              f"mean={vals.mean():+.4f} std={vals.std():.4f} "
              f"max=|{vals.abs().max():.4f}|")

    # 检查是否有全零层（说明没学到东西）
    zero_layers = [n for n, p in net.named_parameters()
                   if p.abs().max() < 1e-6]
    if zero_layers:
        print(f"\n⚠️ 存在近似全零的层: {zero_layers} — 模型可能没被正确训练")
    else:
        print("\n✅ 所有层权重非零，看起来有正常训练痕迹")


if __name__ == '__main__':
    main()
