"""
CAM (原始版) 对 FIA 攻击前后图像的热力图对比分析
=================================================

原始 CAM (Zhou et al., CVPR 2016, "Learning Deep Features for Discriminative
Localization", https://arxiv.org/abs/1512.04150) 要求网络在卷积特征图之后
紧跟**全局平均池化层 (GAP)** 再连分类全连接层。这样 fc 的权重 W[c,k] 就直接
成为第 k 个通道对类别 c 的激活权重:

    cam_c = ReLU( Σ_k W[c,k] · A_k )

其中 A 是目标卷积层的特征图 (C, H, W)。

本脚本流程:
  1. 训练/加载一个带 GAP 的 CIFAR-10 模型 (CIFARGAPNet);
  2. 用 code/fia_attack.py 的 FIA (Feature Importance-aware Attack) 生成对抗样本;
  3. 用原始 CAM 对 "干净图像" 与 "FIA 攻击后图像" 分别生成热力图;
  4. 前后对比: 预测类别/置信度是否改变, 热力图高亮区域如何漂移/被抑制。

FIA 的核心思想正是"抑制模型依赖的判别特征", 因此攻击成功后 CAM 的热区通常会
明显减弱或迁移 —— 这是本脚本要展示的核心现象。

用法
----
    python cam_fia_analysis.py --train          # 先训练 GAP 模型 (约 15-25 分钟)
    python cam_fia_analysis.py                  # 加载已训练模型, 跑 FIA + CAM 对比
    python cam_fia_analysis.py --samples 8 --ens 10 --num-iter 5
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)

from fia_attack import FIA  # 复用 code/fia_attack.py 的 FIA 攻击

PICTURE_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", "picture"))
os.makedirs(PICTURE_DIR, exist_ok=True)
MODEL_PATH = os.path.abspath(os.path.join(CODE_DIR, "..", "data", "cifar_gap_net.pth"))

CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')


# ==================== 带 GAP 的 CIFAR-10 模型 ====================

class CIFARGAPNet(nn.Module):
    """
    紧凑 CIFAR-10 CNN, 关键结构: 卷积 → GAP (全局平均池化) → fc。
    GAP 使原始 CAM 成立: fc 权重即类激活权重。
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)   # ← GAP: 原始 CAM 的关键
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 16×16
        x = self.pool(F.relu(self.conv2(x)))   # 8×8
        x = F.relu(self.conv3(x))              # 64×8×8
        x = self.gap(x).flatten(1)             # 64
        x = self.fc(x)                         # logits (10)
        return x


def load_model(device):
    model = CIFARGAPNet()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device,
                                         weights_only=True))
        print(f"  GAP 模型已加载: {MODEL_PATH}")
    else:
        raise FileNotFoundError(
            f"未找到模型权重 {MODEL_PATH},请先运行: "
            f"python cam_fia_analysis.py --train")
    return model.to(device).eval()


def train_model(device, epochs=25, batch_size=256, lr=1e-3):
    """离线训练 CIFARGAPNet (CPU 上约 15-25 分钟)。"""
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root="../data", train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root="../data", train=False, download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = CIFARGAPNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[训练] CIFARGAPNet, epochs={epochs}, batch={batch_size}, lr={lr}")
    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running = 0, 0, 0.0
        for imgs, labs in trainloader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labs)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labs).sum().item()
            total += imgs.size(0)
        scheduler.step()
        # 每 5 个 epoch 报一次测试准确率
        if epoch % 2 == 0 or epoch == epochs:
            acc = correct / total * 100
            tacc = test_accuracy(model, testloader, device)
            print(f"  epoch {epoch:2d}/{epochs}  train_acc {acc:5.1f}%  "
                  f"test_acc {tacc:5.1f}%")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"  ✅ 模型已保存: {MODEL_PATH}")
    return model


def test_accuracy(model, testloader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labs in testloader:
            out = model(imgs.to(device))
            correct += (out.argmax(1) == labs).sum().item()
            total += labs.size(0)
    return correct / total * 100


# ==================== 原始 CAM ====================

class CAM:
    """
    原始 Class Activation Mapping。

    要求: 目标层输出特征图 A 之后直接接 GAP, 再接分类 fc 层。
    公式: cam_c = ReLU( Σ_k W_fc[c,k] · A_k )

    Parameters:
        model: 已 eval 的分类模型
        target_layer: 目标卷积层 (输出 A)
        fc: 分类全连接层 (weight 形状 (num_classes, C), C = A 的通道数)
    """

    def __init__(self, model, target_layer, fc):
        self.model = model.eval()
        self.fc = fc
        self._activation = None
        self._handle = target_layer.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        self._activation = out

    def __call__(self, x, class_idx=None):
        out = self.model(x)
        A = self._activation  # (B, C, h, w)
        if class_idx is None:
            class_idx = out.argmax(dim=1)
        # 取目标类别对应的 fc 权重行
        W = self.fc.weight[class_idx]  # (B, C)
        cam = torch.einsum("bc,bchw->bhw", W, A)  # (B, h, w)
        cam = F.relu(cam)

        # 逐样本 min-max 归一化
        B = cam.size(0)
        cam_flat = cam.view(B, -1)
        cmin = cam_flat.min(dim=1, keepdim=True)[0]
        cmax = cam_flat.max(dim=1, keepdim=True)[0]
        cam_flat = (cam_flat - cmin) / (cmax - cmin + 1e-8)
        cam = cam_flat.view(B, A.size(2), A.size(3))

        # 上采样到输入尺寸
        cam = F.interpolate(cam[:, None], size=x.shape[-2:],
                            mode="bilinear", align_corners=False)
        return cam.detach(), out.detach(), class_idx

    def remove(self):
        self._handle.remove()


# ==================== 数据与可视化 ====================

def load_test_samples(num_samples, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.CIFAR10(
        root="../data", train=False, download=False, transform=transform)
    testset = torch.utils.data.Subset(testset, list(range(num_samples)))
    loader = torch.utils.data.DataLoader(testset, batch_size=num_samples, shuffle=False)
    images, labels = next(iter(loader))
    return images.to(device), labels


def denorm(img):
    """CIFAR-10 归一化 (mean=std=0.5) → [0,1] 显示。"""
    return (img * 0.5 + 0.5).clamp(0, 1)


def visualize_compare(model, clean, adv, labels, cam_layer, args):
    """对每张样本绘制 [原图+CAM | 攻击图+CAM] 对比,并打印预测变化。"""
    device = next(model.parameters()).device
    cam = CAM(model, cam_layer, model.fc)
    n = clean.shape[0]

    with torch.no_grad():
        clean_out = model(clean)
        adv_out = model(adv)
    clean_pred = clean_out.argmax(1)
    adv_pred = adv_out.argmax(1)
    clean_conf = torch.softmax(clean_out, 1).max(1).values
    adv_conf = torch.softmax(adv_out, 1).max(1).values

    # 每个样本的 CAM: 解释"模型当前预测的类别" (clean 用 clean 预测, adv 用 adv 预测)
    cam_clean, _, _ = cam(clean, class_idx=clean_pred)
    cam_adv, _, _ = cam(adv, class_idx=adv_pred)
    cam.remove()

    fig, axes = plt.subplots(n, 4, figsize=(14, 3.2 * n))
    for i in range(n):
        row = axes[i] if n > 1 else axes
        true_c = CIFAR_CLASSES[labels[i].item()]
        pc = CIFAR_CLASSES[clean_pred[i].item()]
        pa = CIFAR_CLASSES[adv_pred[i].item()]
        flipped = (clean_pred[i] != adv_pred[i]).item()

        # 原图 + CAM
        row[0].imshow(denorm(clean[i]).permute(1, 2, 0))
        row[0].set_title(f"原图\n真值 {true_c}\n预测 {pc}\nP={clean_conf[i]:.2f}",
                         fontsize=9)
        row[0].axis("off")
        row[1].imshow(denorm(clean[i]).permute(1, 2, 0))
        row[1].imshow(cam_clean[i, 0].cpu(), cmap="jet", alpha=0.5)
        row[1].set_title("CAM (攻击前)", fontsize=9)
        row[1].axis("off")

        # 对抗图 + CAM
        row[2].imshow(denorm(adv[i]).permute(1, 2, 0))
        color = "red" if flipped else "green"
        row[2].set_title(f"FIA 对抗图\n预测 {pa} {'✗ 已翻错' if flipped else '✓ 未翻'}"
                         f"\nP={adv_conf[i]:.2f}", fontsize=9, color=color)
        row[2].axis("off")
        row[3].imshow(denorm(adv[i]).permute(1, 2, 0))
        row[3].imshow(cam_adv[i, 0].cpu(), cmap="jet", alpha=0.5)
        row[3].set_title("CAM (攻击后)", fontsize=9, color=color)
        row[3].axis("off")

    fig.suptitle("原始 CAM: FIA 攻击前后热力图对比", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(PICTURE_DIR, "cam_fia_compare.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 已保存: {save_path}")

    # 文字汇总
    print("\n" + "=" * 72)
    print("  前后对比统计")
    print("=" * 72)
    for i in range(n):
        true_c = CIFAR_CLASSES[labels[i].item()]
        pc = CIFAR_CLASSES[clean_pred[i].item()]
        pa = CIFAR_CLASSES[adv_pred[i].item()]
        flipped = (clean_pred[i] != adv_pred[i]).item()
        mark = "✗ 翻错" if flipped else "✓ 未翻"
        print(f"  [{i}] 真值 {true_c:6s} | 攻击前: {pc:6s} "
              f"({clean_conf[i]:.2f}) → 攻击后: {pa:6s} "
              f"({adv_conf[i]:.2f}) | {mark}")
    return {"clean_pred": clean_pred, "adv_pred": adv_pred, "labels": labels}


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="原始 CAM + FIA 攻击前后热力图对比")
    parser.add_argument("--train", action="store_true", help="训练 GAP 模型")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--samples", type=int, default=6, help="对比样本数")
    parser.add_argument("--fia-layer", type=str, default="conv3",
                        help="FIA 攻击的目标特征层")
    parser.add_argument("--cam-layer", type=str, default="conv3",
                        help="CAM 的目标特征层 (需后接 GAP)")
    parser.add_argument("--eps", type=float, default=10 / 255)
    parser.add_argument("--num-iter", type=int, default=10)
    parser.add_argument("--ens", type=int, default=20)
    parser.add_argument("--keep-prob", type=float, default=0.9)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72)
    print("  原始 CAM × FIA 攻击前后对比分析")
    print(f"  设备: {device}")
    print("=" * 72)

    if args.train:
        train_model(device, epochs=args.epochs)

    model = load_model(device)

    # 目标层
    cam_layer = dict(model.named_modules())[args.cam_layer]
    fia_layer = args.fia_layer
    print(f"  CAM 目标层: {args.cam_layer} | FIA 目标层: {fia_layer}")

    # 取样本
    clean, labels = load_test_samples(args.samples, device)
    clean_acc = (model(clean).argmax(1) == labels).float().mean().item()
    print(f"  样本数: {args.samples}, 模型对干净样本识别正确率: {clean_acc * 100:.1f}%")

    # FIA 攻击
    print("\n[FIA 攻击] ...")
    attacker = FIA(model, layer_name=fia_layer, eps=args.eps,
                   alpha=args.eps / args.num_iter, num_iter=args.num_iter,
                   momentum=1.0, ens=args.ens, keep_prob=args.keep_prob,
                   device=device)
    adv = attacker.forward(clean)
    print(f"  攻击完成, 扰动 L∞ = {(adv - clean).abs().max():.4f} "
          f"(预算 {args.eps:.4f})")

    # CAM 前后对比
    visualize_compare(model, clean, adv, labels, cam_layer, args)

    print("\n✅ 分析完成")


if __name__ == "__main__":
    main()
