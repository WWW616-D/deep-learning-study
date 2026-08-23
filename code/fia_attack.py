"""
FIA: Feature Importance-aware Attack (PyTorch 移植版)
=======================================================

参考代码: FIA (Feature Importance-aware) — 通过随机遮挡输入计算特征重要性权重，
然后沿"使特征图加权和最大化"的方向迭代生成对抗样本。

核心思想 (参考 TensorFlow 原版):
  1. 用 ens 个随机遮挡 (dropout mask) 的图像计算特征图的重要性权重:
        weights = ∂(logits · one_hot_label) / ∂(特征图)   在遮挡输入上求
     多遮挡取平均，再 L2 归一化并取负号 (FIA 公式)。
  2. 攻击损失:
        loss = Σ(adv特征图 · weights) / 特征图元素数
     最大化该损失 → 抑制模型依赖的判别特征。
  3. 动量迭代 (MI) 更新:
        noise = momentum · prev_noise + grad / mean(|grad|)
        adv += alpha · sign(noise)

用法:
    python fia_attack.py --model cifar10 --layer conv2 --num-iter 10
    python fia_attack.py --model cifar10 --layer conv2 --ens 10 --num-iter 5  # 快速测试
    python fia_attack.py --eval-only  # 仅用已保存的对抗样本评估
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from gd_uap import CIFAR10Net, MNISTNet, load_model_flexible, evaluate_uap

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

PICTURE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'picture'))
os.makedirs(PICTURE_DIR, exist_ok=True)

CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck')


# ==================== FIA 攻击实现 ====================

class FIA:
    """
    Feature Importance-aware Attack

    Parameters:
        model: 目标模型
        layer_name: 攻击的目标特征层名 (如 CIFAR10Net 的 'conv2')
        eps: 扰动预算 (像素空间 [0,1])
        alpha: 每步迭代步长
        num_iter: 迭代次数
        momentum: 动量衰减系数
        ens: 随机遮挡次数 (计算特征重要性)
        keep_prob: 遮挡保留概率 (1 - 丢弃率)
        device: 计算设备
    """

    def __init__(self, model, layer_name='conv2',
                 eps=10/255, alpha=1.0/255, num_iter=10, momentum=1.0,
                 ens=30, keep_prob=0.9, device='cpu'):
        self.model = model
        self.model.eval()
        self.layer_name = layer_name
        self.eps = eps
        self.alpha = alpha
        self.num_iter = num_iter
        self.momentum = momentum
        self.ens = ens
        self.keep_prob = keep_prob
        self.device = device

        # 注册前向钩子捕获目标层特征图
        self._features = {}
        self._handle = None
        self._target_module = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                self._target_module = module
                self._handle = module.register_forward_hook(
                    self._capture)
                break
        if self._handle is None:
            raise ValueError(f"找不到目标层: {layer_name}")

    def _capture(self, module, inp, out):
        """捕获目标层特征图 (直接赋值，覆盖旧值)"""
        self._features['feat'] = out

    def _remove_hooks(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _compute_weights(self, x):
        """
        计算特征重要性权重 (FIA 核心)

        对 ens 个随机遮挡的输入，求 logits·one_hot 对特征图的梯度，
        取平均后做 L2 归一化并取负号。

        Returns:
            weights: 与目标层特征图同形状的权重张量
        """
        # one-hot 标签: 用模型的预测作为伪标签 (与 TF 版一致)
        with torch.no_grad():
            logits = self.model(x)
            labels = logits.argmax(dim=1)
            num_classes = logits.shape[1]
        one_hot = F.one_hot(labels, num_classes).float()

        sum_grad = None
        n = x.shape[0]

        for _ in range(int(self.ens)):
            # 随机遮挡 (dropout mask)
            mask = torch.from_numpy(
                np.random.binomial(1, self.keep_prob, size=x.shape)
            ).float().to(self.device)
            x_masked = x * mask
            x_masked.requires_grad_(True)

            logits = self.model(x_masked)
            # 确保捕获到本次 forward 的特征图（hook 直接赋值，总会覆盖）
            feat = self._features['feat']
            # 梯度: ∂(logits · one_hot) / ∂特征图
            grads = torch.autograd.grad(
                (logits * one_hot).sum(), feat,
                retain_graph=True)[0]
            if sum_grad is None:
                sum_grad = grads.detach()
            else:
                sum_grad = sum_grad + grads.detach()

        # 平均 → L2 归一化 (每个样本单独) → 取负号
        weights = sum_grad / self.ens
        # 每个样本沿非 batch 维做 L2 归一化
        if weights.dim() == 4:
            norm_dim = (1, 2, 3)
        elif weights.dim() == 2:
            norm_dim = 1
        else:
            norm_dim = tuple(range(1, weights.dim()))
        weights = weights / (weights.norm(p=2, dim=norm_dim, keepdim=True) + 1e-10)
        weights = -weights  # FIA 取负号: 抑制重要特征
        return weights.detach()

    def forward(self, x, verbose=False):
        """
        生成对抗样本

        Args:
            x: 归一化后的输入 (B, C, H, W)
        Returns:
            adv: 对抗样本 (与 x 同形状，归一化空间)
        """
        x = x.to(self.device)
        weights = self._compute_weights(x)
        if verbose:
            print(f"  特征重要性权重: shape={tuple(weights.shape)}, "
                  f"L∞={weights.abs().max().item():.4f}")

        adv = x.clone()
        grad_momentum = torch.zeros_like(x)

        for i in range(self.num_iter):
            adv.requires_grad_(True)
            logits = self.model(adv)
            feat = self._features['feat']

            # FIA 损失: Σ(adv特征图 · weights) / 元素数
            loss = (feat * weights).sum() / feat.numel()

            grad = torch.autograd.grad(loss, adv)[0]
            # 动量更新 (MI)
            grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-10)
            grad_momentum = self.momentum * grad_momentum + grad

            # 更新扰动
            with torch.no_grad():
                adv = adv + self.alpha * grad_momentum.sign()
                # L∞ 约束: 限制在 x 的 eps 邻域内
                adv = torch.clamp(adv, x - self.eps, x + self.eps)
                # 输入范围约束 (归一化后 CIFAR 在 [-1,1])
                adv = torch.clamp(adv, -1, 1)
                adv = adv.detach()

            if verbose:
                print(f"  Iter [{i+1:2d}/{self.num_iter}] "
                      f"FIA Loss: {loss.item():.4f}")

        return adv.detach()


# ==================== 评估与可视化 ====================

def evaluate_fia(model, loader, device, layer_name='conv2', args=None):
    """在数据集上评估 FIA 攻击"""
    model.eval()
    total = 0
    correct_orig = 0
    correct_adv = 0
    fooled = 0
    orig_correct_samples = 0

    attacker = FIA(model, layer_name=layer_name,
                   eps=args.eps, alpha=args.alpha,
                   num_iter=args.num_iter, momentum=args.momentum,
                   ens=args.ens, keep_prob=args.keep_prob,
                   device=device)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.shape[0]

        with torch.no_grad():
            orig_out = model(images)
            orig_preds = orig_out.argmax(dim=1)

        adv = attacker.forward(images)
        with torch.no_grad():
            adv_out = model(adv)
            adv_preds = adv_out.argmax(dim=1)

        correct_orig += (orig_preds == labels).sum().item()
        correct_adv += (adv_preds == labels).sum().item()
        orig_correct_mask = (orig_preds == labels)
        orig_correct_samples += orig_correct_mask.sum().item()
        fooled += ((orig_preds != adv_preds) & orig_correct_mask).sum().item()
        total += batch_size

    orig_acc = correct_orig / total * 100
    adv_acc = correct_adv / total * 100
    fr = fooled / orig_correct_samples * 100 if orig_correct_samples > 0 else 0

    print("\n" + "=" * 60)
    print("           FIA 攻击评估结果")
    print("=" * 60)
    print(f"  测试样本总数:      {total}")
    print(f"  原始准确率:        {orig_acc:.2f}%")
    print(f"  攻击后准确率:      {adv_acc:.2f}%")
    print(f"  Fooling Rate (FR): {fr:.2f}%")
    print(f"  (FR = 原本正确但被扰动攻击成功的样本比例)")
    print(f"  准确率下降:        {orig_acc - adv_acc:.2f}%")
    print("=" * 60)
    return {'orig_acc': orig_acc, 'adv_acc': adv_acc,
            'fooling_rate': fr, 'total': total}


def visualize_attack(model, loader, device, layer_name='conv2',
                     args=None, save_path=None):
    """可视化 FIA 在单张图片上的攻击效果"""
    attacker = FIA(model, layer_name=layer_name,
                   eps=args.eps, alpha=args.alpha,
                   num_iter=args.num_iter, momentum=args.momentum,
                   ens=args.ens, keep_prob=args.keep_prob,
                   device=device)

    images, labels = next(iter(loader))
    img = images[:1].to(device)
    label = labels[:1].to(device)

    # 计算权重 (用于显示特征重要性)
    weights = attacker._compute_weights(img)

    adv = attacker.forward(img)

    with torch.no_grad():
        orig_idx = model(img).argmax(dim=1).item()
        adv_idx = model(adv).argmax(dim=1).item()

    def to_np(t):
        t = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        t = (t - t.min()) / (t.max() - t.min() + 1e-8)
        return np.clip(t, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))

    axes[0].imshow(to_np(img))
    axes[0].set_title(f'Original\n{CIFAR_CLASSES[orig_idx]}', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(to_np(adv - img))
    axes[1].set_title('Perturbation (10x)', fontsize=12)
    axes[1].axis('off')

    # 特征重要性可视化 (取最大通道)
    w = weights[0].detach().cpu()
    w_map = w.norm(dim=0)  # (H, W)
    w_map = (w_map - w_map.min()) / (w_map.max() - w_map.min() + 1e-8)
    axes[2].imshow(w_map.numpy(), cmap='jet')
    axes[2].set_title('Feature Importance', fontsize=12)
    axes[2].axis('off')

    color = 'red' if adv_idx != orig_idx else 'green'
    axes[3].imshow(to_np(adv))
    axes[3].set_title(f'Adversarial\n{CIFAR_CLASSES[adv_idx]}',
                      fontsize=12, color=color)
    axes[3].axis('off')

    plt.suptitle(f'FIA Attack on {layer_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 已保存: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='FIA: Feature Importance-aware Attack')
    parser.add_argument('--model', type=str, default='cifar10',
                        help='目标模型 (默认: cifar10)')
    parser.add_argument('--layer', type=str, default='conv2',
                        help='目标特征层名 (默认: conv2)')
    parser.add_argument('--eps', type=float, default=10/255,
                        help='扰动预算 (默认: 10/255)')
    parser.add_argument('--alpha', type=float, default=1.0/255,
                        help='迭代步长 (默认: 1/255)')
    parser.add_argument('--num-iter', type=int, default=10,
                        help='迭代次数 (默认: 10)')
    parser.add_argument('--momentum', type=float, default=1.0,
                        help='动量系数 (默认: 1.0)')
    parser.add_argument('--ens', type=int, default=30,
                        help='随机遮挡次数 (默认: 30)')
    parser.add_argument('--keep-prob', type=float, default=0.9,
                        help='遮挡保留概率 (默认: 0.9)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='评估批次大小 (默认: 32)')
    parser.add_argument('--num-samples', type=int, default=256,
                        help='评估样本数 (默认: 256, 全量用 10000 很慢)')
    parser.add_argument('--eval-only', action='store_true',
                        help='仅加载已保存的对抗样本评估')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  FIA: Feature Importance-aware Attack")
    print(f"  设备: {device}")
    print(f"  模型: {args.model} | 目标层: {args.layer}")
    print(f"  eps: {args.eps:.4f} | alpha: {args.alpha:.4f}")
    print(f"  迭代: {args.num_iter} | 动量: {args.momentum}")
    print(f"  ens: {args.ens} | keep_prob: {args.keep_prob}")
    print("=" * 60)

    # 加载模型
    model, input_size = load_model_flexible(args.model, device)

    # 加载数据 (CIFAR-10 带归一化, 与训练一致)
    if args.model == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = torchvision.datasets.CIFAR10(
            root='../data', train=False, download=False, transform=transform)
        # 取子集加速
        n = min(args.num_samples, len(testset))
        testset = torch.utils.data.Subset(testset, list(range(n)))
    elif args.model == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = torchvision.datasets.MNIST(
            root='../data', train=False, download=False, transform=transform)
        n = min(args.num_samples, len(testset))
        testset = torch.utils.data.Subset(testset, list(range(n)))
    else:
        raise ValueError(f"FIA 暂只支持 cifar10/mnist: {args.model}")

    loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"评估样本数: {n}")

    # 单张可视化
    visualize_attack(model, loader, device, layer_name=args.layer,
                     args=args, save_path=os.path.join(
                         PICTURE_DIR, 'fia_attack_effect.png'))

    # 批量评估
    results = evaluate_fia(model, loader, device, layer_name=args.layer, args=args)

    # 保存扰动对比图 (FIA vs 原始)
    attacker = FIA(model, layer_name=args.layer,
                   eps=args.eps, alpha=args.alpha,
                   num_iter=args.num_iter, momentum=args.momentum,
                   ens=args.ens, keep_prob=args.keep_prob, device=device)
    images, _ = next(iter(loader))
    img = images[:8].to(device)
    adv = attacker.forward(img)
    pert = adv - img

    # 显示 4 张原始 + 4 张对抗
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i in range(4):
        axes[0, i].imshow((img[i].permute(1, 2, 0).cpu().numpy() + 1) / 2)
        axes[0, i].set_title('Original', fontsize=10)
        axes[0, i].axis('off')
        axes[1, i].imshow((adv[i].permute(1, 2, 0).cpu().numpy() + 1) / 2)
        axes[1, i].set_title('Adversarial', fontsize=10)
        axes[1, i].axis('off')
    plt.suptitle('FIA 攻击对比 (上: 原始, 下: 对抗)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PICTURE_DIR, 'fia_batch_compare.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 已保存: fia_batch_compare.png")

    print("\n✅ FIA 攻击完成！")


if __name__ == '__main__':
    main()
