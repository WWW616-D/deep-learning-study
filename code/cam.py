"""
CAM: Class Activation Mapping — CIFAR-10 图像识别 + 热力图可视化
================================================================

用 Grad-CAM (Gradient-weighted Class Activation Mapping, CVPR 2017,
https://arxiv.org/abs/1610.02391) 解释 CIFAR-10 模型"看哪里"识别图像:

  1. 前向: 把输入送进模型得到 logits,同时用 forward hook 捕获目标卷积层的
     特征图 A (B, C, H, W);
  2. 反向: 对预测类别得分求 A 的梯度 g,再对空间维做全局平均池化得到
     每个通道的重要性权重 α_c;
  3. 加权求和 + ReLU:  cam = ReLU(Σ_c α_c · A_c),归一化后上采样到输入尺寸;
  4. 叠加: 把热力图 (jet) 半透明叠到原图上,输出 "原图 / 热力图 / 叠加"。

实现参考
--------
- transferattack/model_related/ata_vit_utils/Transformer_Explainability/
  baselines/ViT/ViT_explanation_generator.py 中的 CAM 思路
  (用梯度加权注意力图 → ReLU → min-max 归一化)。
- 原版 CAM 需要全局平均池化层 (GAP),而 CIFAR10Net 没有 GAP,
  故采用 Grad-CAM (无需改网络结构)。

用法
----
    python cam.py                                  # CIFAR-10, 自动选最后一个卷积层
    python cam.py --samples 8 --topk 3             # 8 张样本, 显示 Top-3 预测
    python cam.py --layers conv1,conv2             # 对比两个卷积层的热力图
    python cam.py --model mnist                    # 切到 MNIST 模型
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gd_uap import load_model_flexible  # 复用模型加载 (cifar10 / mnist)

PICTURE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "picture"))
os.makedirs(PICTURE_DIR, exist_ok=True)

CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')


# ==================== Grad-CAM 实现 ====================

class GradCAM:
    """
    通用的 Grad-CAM,适用于任意 CNN (含 CIFAR10Net / MNISTNet)。

    Parameters:
        model: 分类模型 (已 eval, 输出 logits 或 log_softmax 均可)
        target_layer: 要做 CAM 的目标层 (nn.Module, 通常是一个 Conv2d)
    """

    def __init__(self, model, target_layer):
        self.model = model.eval()
        self._target_layer = target_layer
        self._activation = None
        self._handle = target_layer.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        self._activation = out

    def __call__(self, x, class_idx=None):
        """
        Args:
            x: (B, C, H, W) 输入 (已归一化)
            class_idx: 每个样本要解释的类别, None 表示用模型自身预测 (Top-1)
        Returns:
            cam:       (B, 1, H, W) 归一化到 [0,1] 的热力图 (与输入同尺寸)
            logits:    模型输出
            class_idx: 实际使用的类别索引
        """
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1)
        A = self._activation  # 目标层特征图 (B, C, h, w)

        # 第 c 类得分的梯度对特征图
        one_hot = torch.zeros_like(out)
        one_hot.scatter_(1, class_idx.unsqueeze(1), 1.0)
        grads = torch.autograd.grad((out * one_hot).sum(), A,
                                    retain_graph=False, create_graph=False)[0]

        # 通道重要性: 梯度全局平均池化
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
        # 加权求和 + ReLU
        cam = torch.relu((A * weights).sum(dim=1, keepdim=True))  # (B,1,h,w)

        # 逐样本 min-max 归一化
        B = cam.size(0)
        cam_flat = cam.view(B, -1)
        cmin = cam_flat.min(dim=1, keepdim=True)[0]
        cmax = cam_flat.max(dim=1, keepdim=True)[0]
        cam_flat = (cam_flat - cmin) / (cmax - cmin + 1e-8)
        cam = cam_flat.view(B, 1, A.size(2), A.size(3))

        # 上采样到输入尺寸
        cam = F.interpolate(cam, size=x.shape[-2:],
                            mode="bilinear", align_corners=False)
        return cam.detach(), out.detach(), class_idx

    def remove(self):
        self._handle.remove()


def find_conv_layers(model):
    """返回模型中所有 Conv2d 层: [(name, module), ...]"""
    import torch.nn as nn
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            layers.append((name, module))
    return layers


# ==================== 数据与可视化 ====================

def load_data(model_name, num_samples, batch_size=8):
    """加载测试集并取前 num_samples 个样本。返回 (images, labels, classes)。"""
    if model_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root="../data", train=False, download=False, transform=transform)
        classes = CIFAR_CLASSES
    elif model_name == "mnist":
        transform = transforms.Compose([transforms.ToTensor()])
        testset = torchvision.datasets.MNIST(
            root="../data", train=False, download=False, transform=transform)
        classes = tuple(str(i) for i in range(10))
    else:
        raise ValueError(f"不支持的数据集: {model_name}")

    testset = torch.utils.data.Subset(testset, list(range(num_samples)))
    loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    images, labels = [], []
    for imgs, labs in loader:
        images.append(imgs)
        labels.append(labs)
    return torch.cat(images), torch.cat(labels), classes


def denorm_cifar(img):
    """CIFAR-10 归一化 (mean=std=0.5) → [0,1] 显示。"""
    return (img * 0.5 + 0.5).clamp(0, 1)


def visualize_samples(model, images, labels, classes, layer_specs, topk,
                      save_path):
    """
    对每个样本生成 [原图 | 热力图 | 叠加] 三列可视化。

    layer_specs: 需要展示 CAM 的目标层名列表 (如 ['conv2'] 或 ['conv1','conv2'])
    """
    images = images.cpu()
    device = next(model.parameters()).device
    B = images.shape[0]
    cols = 3 * len(layer_specs) + 1  # 原图 + 每层各 2 列 (热力图/叠加)

    with torch.no_grad():
        logits = model(images.to(device)).cpu()
    probs = torch.softmax(logits, dim=1) if logits.shape[-1] > 1 else logits.exp()
    topk_idx = probs.topk(topk, dim=1).indices

    fig, axes = plt.subplots(B, cols, figsize=(3.2 * cols, 3.2 * B))

    for i in range(B):
        # 第 1 列: 原图 + 标签/预测
        ax = axes[i, 0] if B > 1 else axes[0]
        ax.imshow(denorm_cifar(images[i]).permute(1, 2, 0))
        true_name = classes[labels[i].item()]
        pred_name = classes[topk_idx[i, 0].item()]
        status = "✓" if labels[i].item() == topk_idx[i, 0].item() else "✗"
        ax.set_title(f"真值 {true_name}\n预测 {pred_name} {status}\n"
                     f"P={probs[i, topk_idx[i, 0]]:.2f}", fontsize=10)
        ax.axis("off")

        col = 1
        for layer_name in layer_specs:
            target = dict((n, m) for n, m in find_conv_layers(model))[layer_name]
            gradcam = GradCAM(model, target)
            # 解释预测的 Top-1 类别
            cls = torch.tensor([topk_idx[i, 0].item()], device=device)
            cam, _, _ = gradcam(images[i:i + 1].to(device), class_idx=cls)
            gradcam.remove()
            cam = cam[0, 0].cpu().numpy()  # (H, W)

            # 热力图 (jet)
            ax_h = axes[i, col] if B > 1 else axes[col]
            im = ax_h.imshow(cam, cmap="jet", vmin=0, vmax=1)
            ax_h.set_title(f"{layer_name} 热力图", fontsize=10)
            ax_h.axis("off")

            # 叠加 (原图 + 半透明热力图)
            ax_o = axes[i, col + 1] if B > 1 else axes[col + 1]
            ax_o.imshow(denorm_cifar(images[i]).permute(1, 2, 0))
            ax_o.imshow(cam, cmap="jet", alpha=0.45)
            ax_o.set_title(f"{layer_name} 叠加", fontsize=10)
            ax_o.axis("off")
            col += 2

    fig.suptitle(f"Grad-CAM 热力图 ({len(layer_specs)} 层, 解释 Top-1 预测)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ 已保存: {save_path}")


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="CAM: CIFAR-10 识别 + 热力图")
    parser.add_argument("--model", type=str, default="cifar10",
                        help="cifar10 / mnist")
    parser.add_argument("--layers", type=str, default=None,
                        help="目标卷积层名, 逗号分隔 (默认: 最后一个卷积层)")
    parser.add_argument("--samples", type=int, default=6, help="样本数")
    parser.add_argument("--topk", type=int, default=1, help="显示 Top-K 预测")
    parser.add_argument("--out", type=str, default=None,
                        help="输出图片路径 (默认 picture/cam_samples.png)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("  Grad-CAM: CIFAR 图像识别 + 热力图")
    print(f"  设备: {device} | 模型: {args.model} | 样本: {args.samples}")
    print("=" * 70)

    # 加载模型
    model, _ = load_model_flexible(args.model, device)
    model.eval()

    # 选择目标层
    conv_layers = find_conv_layers(model)
    if not conv_layers:
        raise ValueError("模型中找不到 Conv2d 层,无法做 CAM")
    if args.layers:
        layer_specs = [l.strip() for l in args.layers.split(",") if l.strip()]
        names = [n for n, _ in conv_layers]
        for l in layer_specs:
            if l not in names:
                raise ValueError(f"层 {l} 不在模型中,可用: {names}")
    else:
        layer_specs = [conv_layers[-1][0]]  # 默认最后一个卷积层
    print(f"  CAM 目标层: {layer_specs}")

    # 加载数据
    images, labels, classes = load_data(args.model, args.samples)
    print(f"  测试样本: {args.samples} 张")

    # 生成热力图
    save_path = args.out or os.path.join(PICTURE_DIR, "cam_samples.png")
    visualize_samples(model, images, labels, classes, layer_specs,
                      args.topk, save_path)

    print("\n✅ CAM 可视化完成")


if __name__ == "__main__":
    main()
