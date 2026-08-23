"""
GD-UAP 不同 Prior 在深度分类模型 (resnet18) 上的对比实验
==========================================================

由于环境没有 ImageNet 真实数据，本实验用以下指标评估 prior 的攻击力:
  1. 对高斯输入的 logits 扰动幅度（L2 范数）— 反映扰动"破坏力"
  2. 对 CIFAR-10 测试集（ImageNet 归一化）的预测改变率（Prediction Change Rate）
  3. 单张真实图像攻击效果可视化

生成的图表（保存到 D:\\py\\picture\\）:
    - 10_deep_loss_curves.png   : 4 种 prior 的训练损失曲线
    - 11_deep_logits_disrupt.png: 扰动后 logits 变化幅度对比
    - 12_deep_attack_effect.png : 真实图像攻击效果对比
    - 13_deep_perturbations.png : 4 种 prior 生成的扰动可视化

用法:
    python prior_experiment_deep.py              # 默认 resnet18, 800 迭代
    python prior_experiment_deep.py --max-iter 500
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from gd_uap import GDUAP, load_model_flexible

PICTURE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'picture'))
os.makedirs(PICTURE_DIR, exist_ok=True)

PRIORS = ['black', 'range', 'gaussian', 'jigsaw']
PRIOR_LABELS = {
    'black': 'Black-image', 'range': 'Range-prior',
    'gaussian': 'Gaussian-prior', 'jigsaw': 'Jigsaw-prior',
}
PRIOR_COLORS = {
    'black': '#444444', 'range': '#e67e22',
    'gaussian': '#2980b9', 'jigsaw': '#27ae60',
}

# ImageNet 归一化（resnet18 等预训练模型要求）
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(images):
    """ImageNet 归一化"""
    return (images - IMAGENET_MEAN) / IMAGENET_STD


def logits_disruption(model, v, n_samples=100, device='cpu'):
    """
    计算扰动对随机高斯输入的 logits 破坏程度

    指标: E_x[ ||f(x+v) - f(x)||_2 ]
    """
    model.eval()
    v = v.to(device)
    total_diff = 0.0
    changed = 0
    with torch.no_grad():
        for _ in range(n_samples):
            x = torch.randn((1, *v.shape[1:]), device=device) * 0.5
            o1 = model(normalize(torch.clamp(x, 0, 1)))
            o2 = model(normalize(torch.clamp(x + v, 0, 1)))
            total_diff += (o1 - o2).norm().item()
            if o1.argmax() != o2.argmax():
                changed += 1
    return total_diff / n_samples, changed / n_samples


def cifar_prediction_change_rate(model, v, device='cpu', n_batches=20):
    """
    在 CIFAR-10 测试集上计算预测改变率

    用 ImageNet 归一化评估（CIFAR 类别与 ImageNet 不匹配，准确率无意义，
    但"预测是否改变"仍能反映扰动的破坏力）。
    """
    import torchvision.datasets as datasets

    testset = datasets.CIFAR10(root='../data', train=False, download=False,
                               transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=0)
    v = v.to(device)

    total = 0
    changed = 0
    for i, (images, _) in enumerate(loader):
        if i >= n_batches:
            break
        images = images.to(device)
        with torch.no_grad():
            o1 = model(normalize(images))
            o2 = model(normalize(torch.clamp(images + v, 0, 1)))
        changed += (o1.argmax(dim=1) != o2.argmax(dim=1)).sum().item()
        total += images.size(0)
    return changed / total * 100


def main():
    parser = argparse.ArgumentParser(
        description='GD-UAP Prior 深度模型对比实验')
    parser.add_argument('--model', type=str, default='resnet18',
                        help='深度分类模型 (默认: resnet18)')
    parser.add_argument('--eps', type=float, default=10/255)
    parser.add_argument('--max-iter', type=int, default=800,
                        help='最大迭代次数 (默认: 800)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='学习率 (默认: 0.1, 大模型用)')
    parser.add_argument('--noise-batch', type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"  GD-UAP 不同 Prior 在深度模型 {args.model} 上的对比实验")
    print(f"  设备: {device} | 迭代: {args.max_iter} | lr: {args.lr}")
    print("=" * 70)

    model, input_size = load_model_flexible(args.model, device)

    # 对每种 prior 生成扰动并评估
    results = {}
    for prior in PRIORS:
        print(f"\n▶ 生成 prior = {PRIOR_LABELS[prior]} ...")
        gduap = GDUAP(
            model, eps=args.eps, max_iter=args.max_iter, lr=args.lr,
            input_size=input_size, noise_batch=args.noise_batch,
            prior=prior, device=device,
        )
        v = gduap.generate(verbose=False)
        v = v.squeeze(0)

        # 评估指标 1: logits 扰动幅度
        disruption, change_prob = logits_disruption(
            model, v, n_samples=50, device=device)

        # 评估指标 2: CIFAR 预测改变率
        pcr = cifar_prediction_change_rate(
            model, v, device=device, n_batches=20)

        results[prior] = {
            'gduap': gduap,
            'perturbation': v,
            'disruption': disruption,
            'change_prob': change_prob * 100,
            'pcr': pcr,
        }
        print(f"    logits 扰动幅度: {disruption:.3f} | "
              f"随机输入改变率: {change_prob*100:.1f}% | "
              f"CIFAR 预测改变率: {pcr:.2f}%")

    # 汇总表
    print("\n" + "=" * 70)
    print(f"  实验结果汇总 (模型: {args.model})")
    print("=" * 70)
    print(f"  {'Prior':<16}{'logitsΔ':>10}{'输入改变率':>12}{'CIFAR PCR':>12}")
    print("-" * 70)
    for prior in PRIORS:
        r = results[prior]
        print(f"  {PRIOR_LABELS[prior]:<16}{r['disruption']:>10.2f}"
              f"{r['change_prob']:>11.1f}%{r['pcr']:>11.2f}%")
    print("=" * 70)

    best = max(PRIORS, key=lambda p: results[p]['disruption'])
    print(f"\n  🏆 扰动最强 prior: {PRIOR_LABELS[best]} "
          f"(logitsΔ={results[best]['disruption']:.2f})")

    # ---- 生成图表 ----
    print("\n生成图表...")

    # 1. 训练损失曲线
    fig, ax = plt.subplots(figsize=(9, 6))
    for prior in PRIORS:
        loss = results[prior]['gduap'].loss_history
        ax.plot(loss, label=PRIOR_LABELS[prior],
                color=PRIOR_COLORS[prior], linewidth=1.2)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Layer-wise Loss')
    ax.set_title(f'GD-UAP 训练损失曲线 ({args.model}, 不同 Prior)',
                 fontsize=13, fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(PICTURE_DIR, '10_deep_loss_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 已保存: 10_deep_loss_curves.png")

    # 2. logits 扰动幅度柱状图
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (metric, title, fmt) in enumerate([
            ('disruption', 'logits 扰动幅度 (L2)', '%.1f'),
            ('change_prob', '随机输入预测改变率 (%)', '%.1f%%'),
            ('pcr', 'CIFAR 预测改变率 (%)', '%.2f%%')]):
        ax = axes[idx]
        vals = [results[p][metric] for p in PRIORS]
        bars = ax.bar(range(4), vals,
                      color=[PRIOR_COLORS[p] for p in PRIORS], width=0.6)
        ax.bar_label(bars, fmt=fmt, fontsize=10)
        ax.set_xticks(range(4))
        ax.set_xticklabels([PRIOR_LABELS[p] for p in PRIORS],
                           rotation=15, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'不同 Prior 的 GD-UAP 攻击力对比 ({args.model})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PICTURE_DIR, '11_deep_metrics.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 已保存: 11_deep_metrics.png")

    # 3. 单张真实图像攻击效果
    test_img = '../data/test_dog.jpg'
    if os.path.exists(test_img):
        transform = transforms.Compose([
            transforms.Resize(input_size[1:]),
            transforms.ToTensor(),
        ])
        img = Image.open(test_img).convert('RGB')
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            orig_idx = model(normalize(img_t)).argmax(dim=1).item()

        fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

        orig_np = img_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        axes[0].imshow(np.clip(orig_np, 0, 1))
        axes[0].set_title(f'Original\nClass {orig_idx}', fontsize=12)
        axes[0].axis('off')

        for i, prior in enumerate(PRIORS):
            v = results[prior]['perturbation'].to(device)
            adv_t = torch.clamp(img_t + v, 0, 1)
            with torch.no_grad():
                adv_idx = model(normalize(adv_t)).argmax(dim=1).item()
            adv_np = adv_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            success = adv_idx != orig_idx
            color = 'red' if success else 'green'
            axes[i+1].imshow(np.clip(adv_np, 0, 1))
            axes[i+1].set_title(
                f'{PRIOR_LABELS[prior]}\nClass {adv_idx} '
                f'{"❌攻击成功" if success else "✅未改变"}',
                fontsize=11, color=color)
            axes[i+1].axis('off')

        plt.suptitle(f'单张图像攻击效果 ({args.model}, 不同 Prior)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        fig.savefig(os.path.join(PICTURE_DIR, '12_deep_attack_effect.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("  ✅ 已保存: 12_deep_attack_effect.png")

    # 4. 扰动可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for i, prior in enumerate(PRIORS):
        v = results[prior]['perturbation']
        p = v.cpu().permute(1, 2, 0).numpy()
        p = (p - p.min()) / (p.max() - p.min() + 1e-8)
        axes[i].imshow(np.clip(p, 0, 1))
        axes[i].set_title(PRIOR_LABELS[prior], fontsize=12)
        axes[i].axis('off')
    plt.suptitle(f'生成的通用对抗扰动 ({args.model})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PICTURE_DIR, '13_deep_perturbations.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  ✅ 已保存: 13_deep_perturbations.png")

    print(f"\n✅ 深度模型对比实验完成！图表在: {PICTURE_DIR}")


if __name__ == '__main__':
    main()
