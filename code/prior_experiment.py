"""
GD-UAP 不同人工图像样本（Prior）对比实验
=========================================

对比 4 种人工图像样本对 GD-UAP 攻击性能的影响:
    - Black-image   : 全黑图像
    - Range-prior   : 均匀分布噪声
    - Gaussian-prior: 高斯噪声
    - Jigsaw-prior  : 拼图打乱的图像

评估指标（在 CIFAR-10 测试集上）:
    - 原始准确率  (Orig Acc)
    - 攻击后准确率 (Adv Acc)
    - Fooling Rate (FR): 原本正确但被扰动攻击成功的样本比例
    - 准确率下降  (Acc Drop)

生成的图表（保存到 D:\\py\\picture\\）:
    - 01_prior_samples.png   : 4 种人工图像样本可视化
    - 02_loss_curves.png     : 训练损失曲线对比
    - 03_perturbations.png   : 4 种 prior 生成的扰动可视化
    - 04_attack_metrics.png  : 攻击指标柱状图对比
    - 05_attack_effect.png   : 单张图片攻击效果对比

用法:
    python prior_experiment.py                # 默认用 CIFAR-10 模型
    python prior_experiment.py --model cifar10
    python prior_experiment.py --max-iter 500 # 减少迭代（快速测试）
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # 无窗口后端，只保存图片
import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from gd_uap import (GDUAP, load_model_flexible, load_data,
                    evaluate_uap, evaluate_uap_single_image,
                    _load_imagenet_labels)

# 结果保存目录
PICTURE_DIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'picture'))
os.makedirs(PICTURE_DIR, exist_ok=True)

# 4 种 prior 配置
PRIORS = ['black', 'range', 'gaussian', 'jigsaw']
PRIOR_LABELS = {
    'black': 'Black-image',
    'range': 'Range-prior',
    'gaussian': 'Gaussian-prior',
    'jigsaw': 'Jigsaw-prior',
}
PRIOR_COLORS = {
    'black': '#444444',
    'range': '#e67e22',
    'gaussian': '#2980b9',
    'jigsaw': '#27ae60',
}


# ==================== 可视化函数 ====================

def plot_prior_samples(gduap_dict, save_path):
    """可视化 4 种人工图像样本"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, prior in zip(axes, PRIORS):
        g = gduap_dict[prior]['gduap']
        samples = g._make_prior_inputs()  # (B, C, H, W)
        img = samples[0].detach().cpu()
        # 归一化到 [0,1] 显示
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = img.permute(1, 2, 0).numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(PRIOR_LABELS[prior], fontsize=13, fontweight='bold')
        ax.axis('off')
    plt.suptitle('人工图像样本 (Artificial Image Priors)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ 已保存: {save_path}")


def plot_loss_curves(gduap_dict, save_path):
    """对比 4 种 prior 的训练损失曲线"""
    fig, ax = plt.subplots(figsize=(9, 6))
    for prior in PRIORS:
        loss = gduap_dict[prior]['gduap'].loss_history
        ax.plot(loss, label=PRIOR_LABELS[prior],
                color=PRIOR_COLORS[prior], linewidth=1.2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Layer-wise Loss (负值越大越好)', fontsize=12)
    ax.set_title('GD-UAP 训练损失曲线对比（不同 Prior）',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ 已保存: {save_path}")


def plot_perturbations(gduap_dict, save_path):
    """可视化 4 种 prior 生成的扰动（放大显示）"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, prior in zip(axes, PRIORS):
        v = gduap_dict[prior]['perturbation']  # (C, H, W)
        img = v.cpu().permute(1, 2, 0).numpy()
        # 放大 10 倍便于观察
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"{PRIOR_LABELS[prior]}\n"
                     f"L∞={gduap_dict[prior]['perturbation'].abs().max().item():.3f}",
                     fontsize=11)
        ax.axis('off')
    plt.suptitle('生成的通用对抗扰动 (10x 归一化显示)',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ 已保存: {save_path}")


def plot_attack_metrics(gduap_dict, save_path):
    """对比 4 种 prior 的攻击指标（柱状图）"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 提取指标
    fr = [gduap_dict[p]['fooling_rate'] for p in PRIORS]
    acc_drop = [gduap_dict[p]['acc_drop'] for p in PRIORS]
    adv_acc = [gduap_dict[p]['adv_acc'] for p in PRIORS]

    x = np.arange(len(PRIORS))
    labels = [PRIOR_LABELS[p] for p in PRIORS]

    # 1. Fooling Rate
    ax = axes[0]
    bars = ax.bar(x, fr, color=[PRIOR_COLORS[p] for p in PRIORS], width=0.6)
    ax.bar_label(bars, fmt='%.1f%%', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=10)
    ax.set_ylabel('Fooling Rate (%)', fontsize=11)
    ax.set_title('Fooling Rate（越高越好）', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(fr) * 1.25 + 1)
    ax.grid(axis='y', alpha=0.3)

    # 2. 准确率下降
    ax = axes[1]
    bars = ax.bar(x, acc_drop, color=[PRIOR_COLORS[p] for p in PRIORS], width=0.6)
    ax.bar_label(bars, fmt='%.1f%%', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=10)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=11)
    ax.set_title('准确率下降（越高越好）', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(acc_drop) * 1.25 + 1)
    ax.grid(axis='y', alpha=0.3)

    # 3. 攻击后准确率
    ax = axes[2]
    bars = ax.bar(x, adv_acc, color=[PRIOR_COLORS[p] for p in PRIORS], width=0.6)
    ax.bar_label(bars, fmt='%.1f%%', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, fontsize=10)
    ax.set_ylabel('Adv Accuracy (%)', fontsize=11)
    ax.set_title('攻击后准确率（越低越好）', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(adv_acc) * 1.25 + 1)
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('不同 Prior 的 GD-UAP 攻击性能对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ 已保存: {save_path}")


def plot_attack_effect(gduap_dict, model, device, save_path):
    """对比 4 种 prior 在单张图片上的攻击效果"""
    # 找一张测试图片
    default_img = '../data/test_dog.jpg'
    if not os.path.exists(default_img):
        print("  ⚠️ 未找到测试图片，跳过攻击效果图")
        return

    from PIL import Image
    import torchvision.transforms as transforms

    # 取其中一个 prior 的扰动尺寸
    v0 = gduap_dict[PRIORS[0]]['perturbation']
    input_size = v0.shape  # (C, H, W)
    n_channels = input_size[0]

    # 根据模型输入通道数转换图像（RGB 或灰度）
    if n_channels == 1:
        convert = 'L'
    else:
        convert = 'RGB'
    transform = transforms.Compose([
        transforms.Resize(input_size[1:]),
        transforms.ToTensor(),
    ])
    image = Image.open(default_img).convert(convert)
    img_tensor = transform(image).unsqueeze(0).to(device)

    # 原始预测
    with torch.no_grad():
        orig_out = model(img_tensor)
        orig_idx = orig_out.argmax(dim=1).item()

    # 每个 prior 的攻击效果
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5))

    # 第一列：原始图像
    orig_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    orig_np = np.clip((orig_np - orig_np.min()) / (orig_np.max() - orig_np.min() + 1e-8), 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title(f'Original\nClass {orig_idx}', fontsize=12)
    axes[0].axis('off')

    # 后续列：每种 prior 的攻击效果
    for i, prior in enumerate(PRIORS):
        v = gduap_dict[prior]['perturbation'].to(device)
        adv_tensor = torch.clamp(img_tensor + v, 0, 1)
        with torch.no_grad():
            adv_out = model(adv_tensor)
            adv_idx = adv_out.argmax(dim=1).item()
        adv_np = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        adv_np = np.clip((adv_np - adv_np.min()) / (adv_np.max() - adv_np.min() + 1e-8), 0, 1)
        success = adv_idx != orig_idx
        color = 'red' if success else 'green'
        axes[i + 1].imshow(adv_np)
        axes[i + 1].set_title(f'{PRIOR_LABELS[prior]}\n'
                              f'Class {adv_idx} '
                              f'{"❌欺骗" if success else "✅未变"}',
                              fontsize=11, color=color)
        axes[i + 1].axis('off')

    plt.suptitle('不同 Prior 的单张图片攻击效果对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ 已保存: {save_path}")


# ==================== 实验主体 ====================

def run_experiment(args):
    print("=" * 70)
    print("  GD-UAP 不同人工图像样本（Prior）对比实验")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  设备: {device}")
    print(f"  模型: {args.model}")
    print(f"  迭代次数: {args.max_iter}")
    print(f"  noise_batch: {args.noise_batch}")
    print(f"  扰动预算 eps: {args.eps}")
    print("=" * 70)

    # ---- 加载模型和测试数据 ----
    model, input_size = load_model_flexible(args.model, device)
    if args.model in ('cifar10', 'mnist'):
        test_loader, classes = load_data(args.model, batch_size=args.batch_size)
        class_names = classes
    else:
        # 大模型：用 ImageNet 验证集子集
        test_loader, class_names = load_data('imagenet_val',
                                             batch_size=args.batch_size)

    # ---- 对每种 prior 生成扰动并评估 ----
    results = {}
    for prior in PRIORS:
        print(f"\n▶ 实验 prior = {PRIOR_LABELS[prior]} ...")
        gduap = GDUAP(
            model,
            eps=args.eps,
            max_iter=args.max_iter,
            lr=args.lr,
            input_size=input_size,
            noise_batch=args.noise_batch,
            prior=prior,
            device=device,
        )
        v = gduap.generate(verbose=False)

        # 数据集评估
        eval_res = evaluate_uap(model, v, test_loader, device, verbose=False)

        results[prior] = {
            'gduap': gduap,
            'perturbation': v.squeeze(0),  # (C, H, W)
            'orig_acc': eval_res['orig_acc'],
            'adv_acc': eval_res['adv_acc'],
            'fooling_rate': eval_res['fooling_rate'],
            'acc_drop': eval_res['orig_acc'] - eval_res['adv_acc'],
        }
        print(f"    Orig Acc: {eval_res['orig_acc']:.2f}%  "
              f"Adv Acc: {eval_res['adv_acc']:.2f}%  "
              f"FR: {eval_res['fooling_rate']:.2f}%")

    # ---- 打印汇总表 ----
    print("\n" + "=" * 70)
    print("  实验结果汇总")
    print("=" * 70)
    print(f"  {'Prior':<16}{'Orig Acc':>10}{'Adv Acc':>10}"
          f"{'FR':>10}{'Acc Drop':>12}")
    print("-" * 70)
    for prior in PRIORS:
        r = results[prior]
        print(f"  {PRIOR_LABELS[prior]:<16}{r['orig_acc']:>9.2f}%"
              f"{r['adv_acc']:>9.2f}%{r['fooling_rate']:>9.2f}%"
              f"{r['acc_drop']:>11.2f}%")
    print("=" * 70)

    # 找出最优 prior
    best = max(PRIORS, key=lambda p: results[p]['fooling_rate'])
    print(f"\n  🏆 攻击性能最优的 prior: {PRIOR_LABELS[best]} "
          f"(FR={results[best]['fooling_rate']:.2f}%)")

    # ---- 保存结果到 CSV ----
    csv_path = os.path.join(PICTURE_DIR, 'prior_experiment_results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('Prior,Orig_Acc,Adv_Acc,Fooling_Rate,Acc_Drop\n')
        for prior in PRIORS:
            r = results[prior]
            f.write(f"{PRIOR_LABELS[prior]},{r['orig_acc']:.2f},"
                    f"{r['adv_acc']:.2f},{r['fooling_rate']:.2f},"
                    f"{r['acc_drop']:.2f}\n")
    print(f"  📊 结果已保存到: {csv_path}")

    # ---- 生成所有图表 ----
    print("\n生成图表...")
    plot_prior_samples(results, os.path.join(PICTURE_DIR, '01_prior_samples.png'))
    plot_loss_curves(results, os.path.join(PICTURE_DIR, '02_loss_curves.png'))
    plot_perturbations(results, os.path.join(PICTURE_DIR, '03_perturbations.png'))
    plot_attack_metrics(results, os.path.join(PICTURE_DIR, '04_attack_metrics.png'))
    plot_attack_effect(results, model, device,
                       os.path.join(PICTURE_DIR, '05_attack_effect.png'))

    print(f"\n✅ 所有图表已保存到: {PICTURE_DIR}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='GD-UAP 不同人工图像样本(Prior)对比实验')
    parser.add_argument('--model', type=str, default='cifar10',
                        help='目标模型 (默认: cifar10)')
    parser.add_argument('--eps', type=float, default=10/255,
                        help='L∞ 扰动预算 (默认: 10/255)')
    parser.add_argument('--max-iter', type=int, default=1500,
                        help='最大迭代次数 (默认: 1500)')
    parser.add_argument('--lr', type=float, default=0.5,
                        help='学习率 (默认: 0.5)')
    parser.add_argument('--noise-batch', type=int, default=4,
                        help='噪声批次大小 (默认: 4)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='评估批次大小 (默认: 64)')
    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()
