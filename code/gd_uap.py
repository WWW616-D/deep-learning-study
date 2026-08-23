"""
GD-UAP: Generalizable Data-free Universal Adversarial Perturbations
===================================================================

基于论文: "Generalizable Data-free Objective for Crafting Universal
           Adversarial Perturbations" (Mopuri et al., 2018)

核心思想：
1. 无需训练数据，仅需预训练模型
2. 通过操纵网络多层激活值来生成通用扰动
3. 生成的扰动具有泛化性：对任意输入都能使模型误分类

支持的模型：
  - 自定义模型: cifar10, mnist
  - torchvision 模型: resnet50, vit_b_16, convnext_tiny, swin_t, ...
    (任意 torchvision.models 中的分类模型)
  - timm 模型: timm:vit_base_patch16_224, timm:convnext_tiny, ...
    (timm.list_models() 中的任意模型)

用法:
    python gd_uap.py                                    # 默认 CIFAR-10
    python gd_uap.py --model resnet50                   # torchvision ResNet-50
    python gd_uap.py --model vit_b_16                   # torchvision ViT-B/16
    python gd_uap.py --model timm:convnext_tiny         # timm ConvNeXt-Tiny
    python gd_uap.py --model resnet152 --noise-batch 8  # 大模型推荐参数
    python gd_uap.py --eval-only                        # 仅评估已保存扰动
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

# 安全导入 timm（如果未安装则提示）
try:
    import timm
    _HAS_TIMM = True
except ImportError:
    _HAS_TIMM = False
    timm = None


# ==================== 工具函数 ====================

def detect_input_size(model):
    """
    自动检测模型的输入尺寸 (C, H, W)

    Returns:
        tuple: (channels, height, width)
    """
    # 1) timm 模型：default_cfg 中通常有 'input_size'
    if hasattr(model, 'default_cfg') and isinstance(model.default_cfg, dict):
        cfg = model.default_cfg
        if 'input_size' in cfg:
            size = cfg['input_size']
            if isinstance(size, (list, tuple)) and len(size) == 3:
                return tuple(size)

    # 2) torchvision ViT：有 image_size 属性
    if hasattr(model, 'image_size'):
        img_size = model.image_size
        if isinstance(img_size, int):
            return (3, img_size, img_size)
        if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            return (3, img_size[0], img_size[1])

    # 3) 传统 CNN：从第一个 Conv2d 推断通道数
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            in_ch = module.weight.shape[1]
            # torchvision ImageNet 模型大多输入 224×224
            # （inception 系列 299×299 由 load_model_flexible 特殊处理）
            hw = 224 if in_ch == 3 else 28
            return (in_ch, hw, hw)

    # 4) 回退
    print("  [Warning] 无法自动检测输入尺寸，使用默认 (3, 224, 224)")
    return (3, 224, 224)


# ==================== 模型定义 ====================

class CIFAR10Net(nn.Module):
    """与 torchtest.py 中一致的 CIFAR-10 模型"""
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


class MNISTNet(nn.Module):
    """与 attack_FGSM.py 中一致的 MNIST LeNet 模型"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# ==================== GD-UAP 核心算法 ====================

class GDUAP:
    """
    Generalizable Data-free Universal Adversarial Perturbation

    Parameters:
        model (nn.Module): 预训练模型
        eps (float): L∞ 范数约束下的最大扰动幅度 (默认 10/255)
        max_iter (int): 最大迭代次数
        lr (float): 学习率 / 迭代步长
        input_size (tuple): 输入图像的形状 (C, H, W)
        prior (str): 人工图像样本类型:
            - 'black'    : 全黑图像 (Black-image)
            - 'range'    : 均匀分布噪声 (Range-prior)
            - 'gaussian' : 高斯噪声 (Gaussian-prior, 默认)
            - 'jigsaw'   : 拼图打乱的图像 (Jigsaw-prior)
        device (torch.device): 计算设备
    """
    def __init__(self, model, eps=10/255, max_iter=2000, lr=0.5,
                 input_size=(3, 224, 224), noise_batch=1,
                 prior='gaussian', device=None):

        self.model = model
        self.model.eval()
        self.eps = eps
        self.max_iter = max_iter
        self.lr = lr
        self.input_size = input_size
        self.noise_batch = noise_batch
        self.prior = prior
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # Jigsaw-prior 的基准图像（首次使用时加载）
        self._jigsaw_base = None

        # 收集目标层名称
        self.activation_layers = self._find_activation_layers()
        self.activations = {}  # 存储各层激活值
        self.handles = []      # hook 句柄

    def _find_activation_layers(self):
        """
        自动查找模型中可用的激活层

        四阶段漏斗式检测，适应不同架构：
          1. ReLU/GELU/SiLU 等激活函数
          2. LayerNorm/GroupNorm（ViT/Swin/ConvNeXt 核心层）
          3. attention 投影 Linear 层
          4. Conv2d 兜底
        最后均匀采样以控制层数，覆盖网络全深度。
        """
        layer_names = []

        # Phase 1: 标准激活层（ReLU family + GELU + SiLU）
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.ReLU, nn.GELU, nn.SiLU,
                                    nn.LeakyReLU, nn.ReLU6)):
                layer_names.append(name)

        # Phase 2: 激活层太少则加入 LayerNorm/GroupNorm
        # （ViT 每个 block 有 2 个 LayerNorm，Swin/ConvNeXt 同理）
        if len(layer_names) < 3:
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
                    layer_names.append(name)

        # Phase 3: 仍然太少则加入 attention 投影层
        if len(layer_names) < 3:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Linear):
                    if any(kw in name.lower() for kw in
                           ['proj', 'qkv', 'fc1', 'fc2', 'mlp']):
                        layer_names.append(name)

        # Phase 4: 最后回退到 Conv2d
        if len(layer_names) == 0:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.Conv2d):
                    layer_names.append(name)

        # 均匀采样以覆盖网络全深度（上限 15 层）
        if len(layer_names) > 15:
            indices = np.linspace(0, len(layer_names) - 1, 15, dtype=int)
            layer_names = [layer_names[i] for i in indices]

        return layer_names

    def _register_hooks(self):
        """注册前向钩子，捕获每层的激活值"""
        self.handles = []
        self.activations = {}

        def get_hook(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook

        for name, module in self.model.named_modules():
            if name in self.activation_layers:
                handle = module.register_forward_hook(get_hook(name))
                self.handles.append(handle)

    def _remove_hooks(self):
        """移除所有钩子"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()

    def _compute_layer_wise_loss(self):
        """
        计算逐层损失函数（GD-UAP 核心目标函数）

        对每层激活值的 L2 范数求和，目标是最大化这个值。
        论文思想：过度激活各层神经元 → 破坏模型正常推理。

        支持不同维度的激活张量：
          - 4D (B,C,H,W): CNN 特征图
          - 3D (B,N,D): Transformer LayerNorm / attention 输出
          - 2D (B,D): 池化/线性层输出
        """
        total_loss = 0.0
        for name in self.activation_layers:
            if name in self.activations:
                act = self.activations[name]
                ndim = act.ndim
                if ndim == 4:
                    # Conv 特征: (B, C, H, W)
                    total_loss += act.norm(p=2, dim=(1, 2, 3)).mean()
                elif ndim == 3:
                    # Transformer 特征: (B, N, D) — 序列/token
                    total_loss += act.norm(p=2, dim=(1, 2)).mean()
                elif ndim == 2:
                    # 池化/线性特征: (B, D)
                    total_loss += act.norm(p=2, dim=1).mean()
                else:
                    # 奇异张量 — flatten 后求 norm
                    total_loss += act.view(act.shape[0], -1).norm(p=2, dim=1).mean()
        return -total_loss  # 最大化 activation → 最小化 -activation

    def _load_jigsaw_base(self):
        """加载 Jigsaw-prior 的基准图像（本地 test_dog.jpg）"""
        from PIL import Image
        base_paths = ['../data/test_dog.jpg', 'data/test_dog.jpg',
                      os.path.join(os.path.dirname(__file__), '..',
                                   'data', 'test_dog.jpg')]
        img = None
        for p in base_paths:
            if os.path.exists(p):
                img = Image.open(p).convert('RGB')
                break
        if img is None:
            # 没有基准图像就退化为随机噪声
            return None
        C, H, W = self.input_size
        img = img.resize((W, H))
        if C == 1:
            # 单通道模型（如 MNIST）：转灰度
            img = img.convert('L')
        arr = np.array(img).astype(np.float32) / 255.0  # [0,1]
        if C == 1:
            arr = arr[..., None]  # (H, W, 1)
        # 中心化到 [-1, 1]，与高斯噪声尺度可比
        arr = (arr - 0.5) * 2.0
        return arr

    def _make_jigsaw_batch(self, n):
        """
        生成 n 个拼图打乱 (jigsaw) 的样本

        将基准图像划分为 p×p 个小块，随机排列，生成结构性伪输入。
        """
        C, H, W = self.input_size
        if self._jigsaw_base is None:
            base = self._load_jigsaw_base()
            if base is None:
                return torch.randn((n, C, H, W), device=self.device)
            self._jigsaw_base = torch.from_numpy(base).permute(2, 0, 1)

        # 拼图网格大小（每个 patch ≥ 8×8）
        p = max(2, min(8, H // 8, W // 8))
        ph, pw = H // p, W // p

        base = self._jigsaw_base.to(self.device)
        patches = base.unfold(1, ph, ph).unfold(2, pw, pw)  # (C, p, p, ph, pw)
        patches = patches.reshape(C, p * p, ph, pw)          # (C, p², ph, pw)
        patches = patches.permute(1, 0, 2, 3)                # (p², C, ph, pw)

        batch = []
        for _ in range(n):
            perm = torch.randperm(p * p, device=self.device)
            shuffled = patches[perm]                         # (p², C, ph, pw)
            shuffled = shuffled.permute(1, 0, 2, 3)          # (C, p², ph, pw)
            img = shuffled.reshape(C, p * ph, p * pw)        # (C, H', W')
            # 尺寸可能因整除损失少量像素，补到输入尺寸
            if img.shape[1:] != (H, W):
                img = F.interpolate(img.unsqueeze(0), size=(H, W),
                                    mode='bilinear', align_corners=False)[0]
            batch.append(img)
        return torch.stack(batch)

    def _make_prior_inputs(self):
        """
        根据 self.prior 生成一批人工图像样本

        Returns:
            torch.Tensor: (noise_batch, C, H, W)
        """
        n = self.noise_batch
        if self.prior == 'black':
            # Black-image: 全黑图像（全零）
            return torch.zeros((n, *self.input_size), device=self.device)
        elif self.prior == 'range':
            # Range-prior: 均匀分布噪声 U(-1, 1)
            return torch.rand((n, *self.input_size), device=self.device) * 2 - 1
        elif self.prior == 'gaussian':
            # Gaussian-prior: 标准高斯噪声 N(0, 1)
            return torch.randn((n, *self.input_size), device=self.device)
        elif self.prior == 'jigsaw':
            # Jigsaw-prior: 拼图打乱的图像
            return self._make_jigsaw_batch(n)
        else:
            raise ValueError(f"未知的 prior 类型: {self.prior}，"
                             f"可选: black, range, gaussian, jigsaw")

    def generate(self, verbose=True):
        """
        生成通用对抗扰动（GD-UAP 主循环）

        关键：输入是随机噪声，不使用任何真实数据！

        Returns:
            v (torch.Tensor): 形状为 (1, C, H, W) 的通用对抗扰动
        """
        if verbose:
            print("=" * 60)
            print("GD-UAP 通用对抗扰动生成")
            print(f"  扰动预算 eps: {self.eps:.4f}")
            print(f"  最大迭代次数: {self.max_iter}")
            print(f"  输入尺寸: {self.input_size}")
            print(f"  噪声批次大小: {self.noise_batch}")
            print(f"  人工图像样本 (prior): {self.prior}")
            print(f"  目标激活层: {len(self.activation_layers)} 层")
            print(f"  设备: {self.device}")
            print("=" * 60)

        # 初始化扰动（全零）
        v = torch.zeros((1, *self.input_size), device=self.device,
                        requires_grad=True)

        # 注册钩子
        self._register_hooks()

        # 使用 SGD 优化器作用于扰动 v
        optimizer = torch.optim.SGD([v], lr=self.lr)

        loss_history = []
        best_loss = float('inf')

        for iteration in range(1, self.max_iter + 1):
            optimizer.zero_grad()

            # 关键：使用人工图像样本 (prior) 作为伪输入，而非真实数据！
            # 每次迭代生成 B 个样本，降低梯度方差
            prior_inputs = self._make_prior_inputs()
            # 将扰动扩展到 batch 维度
            v_batch = v.expand(self.noise_batch, -1, -1, -1)

            # 前向传播（输入 = 人工样本 + 扰动）
            # 注意：我们只需要激活值来计算 loss，不需要真实的 logits
            self.model(prior_inputs + v_batch)

            # 计算 GD-UAP 层间目标函数
            # loss 自动在 noise batch 上取均值
            loss = self._compute_layer_wise_loss()
            loss.backward()

            optimizer.step()

            # 投影到 L∞ 球内
            with torch.no_grad():
                v.clamp_(-self.eps, self.eps)

            loss_val = loss.item()
            loss_history.append(loss_val)

            # 打印训练进度
            if verbose and iteration % 200 == 0:
                print(f"  Iter [{iteration:5d}/{self.max_iter}]  "
                      f"Loss: {loss_val:.6f}  "
                      f"v_norm(L∞): {v.abs().max().item():.6f}")

        # 如果 loss 没有提升，可能是学习率问题
        if verbose:
            print("-" * 60)
            print(f"训练完成！最终 Loss: {loss_history[-1]:.6f}")
            print(f"扰动 L∞ 范数: {v.abs().max().item():.6f}")
            print("=" * 60)

        self._remove_hooks()

        self.loss_history = loss_history
        return v.detach()

    def save(self, v, filepath):
        """保存生成的扰动"""
        torch.save({
            'perturbation': v.cpu(),
            'eps': self.eps,
            'input_size': self.input_size,
            'loss_history': self.loss_history,
        }, filepath)
        print(f"\n扰动已保存到: {filepath}")

    def load(self, filepath):
        """加载已保存的扰动"""
        data = torch.load(filepath, map_location=self.device, weights_only=False)
        v = data['perturbation'].to(self.device)
        self.eps = data['eps']
        self.input_size = data['input_size']
        self.loss_history = data.get('loss_history', [])
        print(f"扰动已加载: {filepath}, eps={self.eps}, shape={v.shape}")
        return v


# ==================== 评估函数 ====================

def _extract_normalize(dataloader):
    """
    从 dataloader 的 dataset transform 中提取归一化参数 (mean, std)

    CIFAR-10 使用 Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) 时返回
    (tensor([0.5,0.5,0.5]), tensor([0.5,0.5,0.5]))，否则返回 (None, None)。

    Returns:
        (mean, std): torch.Tensor 或 None
    """
    try:
        ds = dataloader.dataset
        if hasattr(ds, 'dataset'):
            ds = ds.dataset  # Subset 包装
        transform = getattr(ds, 'transform', None)
        if transform is None:
            return None, None
        # 从 Compose 中找 Normalize
        from torchvision.transforms import Compose, Normalize
        if isinstance(transform, Compose):
            for t in transform.transforms:
                if isinstance(t, Normalize):
                    return (torch.tensor(t.mean, dtype=torch.float32),
                            torch.tensor(t.std, dtype=torch.float32))
        elif isinstance(transform, Normalize):
            return (torch.tensor(transform.mean, dtype=torch.float32),
                    torch.tensor(transform.std, dtype=torch.float32))
    except Exception:
        pass
    return None, None


def evaluate_uap(model, v, dataloader, device, verbose=True):
    """
    评估通用扰动在测试集上的表现

    指标：
        - Fooling Rate (FR): 使模型预测发生改变的比例（越高越好）
        - 原始准确率 vs 攻击后准确率

    Returns:
        dict: 包含评估结果的字典
    """
    model.eval()
    total = 0
    correct_orig = 0
    correct_adv = 0
    fooled = 0  # 原本正确、但加入扰动后错误的样本数
    orig_correct_samples = 0  # 原本就预测正确的样本数

    # 确保扰动在正确的设备上
    v = v.to(device)

    # 检测 dataloader 的归一化方式（如 CIFAR-10 用 Normalize(0.5, 0.5)）
    # 扰动 v 是在原始像素空间 [0,1] 上生成的，评估时需先反归一化再加扰动
    norm_mean, norm_std = _extract_normalize(dataloader)

    for batch in dataloader:
        # 兼容 2 元组 (images, labels) 和 3 元组 (images, labels, filenames)
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.shape[0]

        # 调整扰动大小以匹配输入（如果尺寸不同）
        if v.shape[-2:] != images.shape[-2:]:
            v_resized = F.interpolate(v, size=images.shape[-2:], mode='bilinear')
        else:
            v_resized = v

        # 原始预测
        with torch.no_grad():
            orig_out = model(images)
            orig_preds = orig_out.argmax(dim=1)

            # 对抗预测（加入 UAP）
            if norm_mean is not None:
                # 图像已归一化 → 先反归一化到 [0,1]，加扰动，再归一化
                nm = norm_mean.view(1, -1, 1, 1).to(images.device)
                ns = norm_std.view(1, -1, 1, 1).to(images.device)
                images_raw = images * ns + nm
                adv_raw = torch.clamp(images_raw + v_resized, 0, 1)
                adv_images = (adv_raw - nm) / ns
            else:
                # 图像未归一化 → 直接加扰动
                adv_images = torch.clamp(images + v_resized, 0, 1)
            adv_out = model(adv_images)
            adv_preds = adv_out.argmax(dim=1)

        correct_orig += (orig_preds == labels).sum().item()
        correct_adv += (adv_preds == labels).sum().item()

        # 计算 Fooling Rate：原本正确的样本中，被扰动后变错的比例
        orig_correct_mask = (orig_preds == labels)
        orig_correct_samples += orig_correct_mask.sum().item()
        fooled += ((orig_preds != adv_preds) & orig_correct_mask).sum().item()

        total += batch_size

    orig_acc = correct_orig / total * 100
    adv_acc = correct_adv / total * 100
    fr = fooled / orig_correct_samples * 100 if orig_correct_samples > 0 else 0

    if verbose:
        print("\n" + "=" * 60)
        print("           UAP 评估结果")
        print("=" * 60)
        print(f"  测试样本总数:      {total}")
        print(f"  原始准确率:        {orig_acc:.2f}%")
        print(f"  攻击后准确率:      {adv_acc:.2f}%")
        print(f"  Fooling Rate (FR): {fr:.2f}%")
        print(f"  (FR = 原本正确但被扰动攻击成功的样本比例)")
        print(f"  准确率下降:        {orig_acc - adv_acc:.2f}%")
        print("=" * 60)

    return {
        'orig_acc': orig_acc,
        'adv_acc': adv_acc,
        'fooling_rate': fr,
        'total': total,
    }


def evaluate_uap_single_image(model, v, image_path, device, class_names=None,
                              norm_mean=None, norm_std=None):
    """
    在单张图片上可视化 UAP 攻击效果

    Args:
        norm_mean (list/tensor, optional): 图像归一化均值（如 CIFAR-10 的 0.5）
        norm_std (list/tensor, optional): 图像归一化标准差（如 CIFAR-10 的 0.5）
    """
    model.eval()

    # 使用扰动的尺寸作为模型期望的输入尺寸
    input_size = v.shape[-2:]  # (H, W)

    # 加载并预处理图像（可选归一化）
    if norm_mean is not None:
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
        ])

    image = Image.open(image_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device)

    # 原始预测
    with torch.no_grad():
        out = model(img_tensor)
        orig_idx = out.argmax(dim=1).item()
        orig_conf = torch.softmax(out, dim=1).max().item()

    # 对抗预测（v 在原始像素空间，需反归一化再加扰动）
    if norm_mean is not None:
        nm = torch.tensor(norm_mean).view(1, -1, 1, 1).to(device)
        ns = torch.tensor(norm_std).view(1, -1, 1, 1).to(device)
        raw = img_tensor * ns + nm
        adv_raw = torch.clamp(raw + v, 0, 1)
        adv_tensor = (adv_raw - nm) / ns
    else:
        adv_tensor = torch.clamp(img_tensor + v, 0, 1)
    with torch.no_grad():
        adv_out = model(adv_tensor)
        adv_idx = adv_out.argmax(dim=1).item()
        adv_conf = torch.softmax(adv_out, dim=1).max().item()

    # 转换为 numpy 用于可视化
    orig_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    adv_np = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    pert_np = (adv_np - orig_np) * 10 + 0.5  # 放大 10 倍
    pert_np = np.clip(pert_np, 0, 1)

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    orig_label = class_names[orig_idx] if class_names and orig_idx < len(class_names) else f"Class {orig_idx}"
    adv_label = class_names[adv_idx] if class_names and adv_idx < len(class_names) else f"Class {adv_idx}"

    axes[0].imshow(orig_np)
    axes[0].set_title(f'Original\n{orig_label} ({orig_conf:.2%})', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(pert_np)
    axes[1].set_title('Perturbation (10x amplified)', fontsize=12)
    axes[1].axis('off')

    color = 'red' if adv_idx != orig_idx else 'green'
    axes[2].imshow(adv_np)
    axes[2].set_title(f'Adversarial\n{adv_label} ({adv_conf:.2%})',
                      fontsize=12, color=color)
    axes[2].axis('off')

    plt.suptitle('GD-UAP Attack Result', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(os.path.dirname(__file__) or '.',
                             'gd_uap_result.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    success = adv_idx != orig_idx
    print(f"\n  原始预测: {orig_label} ({orig_conf:.2%})")
    print(f"  对抗预测: {adv_label} ({adv_conf:.2%})")
    print(f"  攻击结果: {'✅ 成功（模型被欺骗）' if success else '❌ 失败（预测未改变）'}")

    return success


# ==================== 模型加载 ====================

def load_model_flexible(model_spec, device):
    """
    加载预训练模型（灵活版）

    支持三种来源:
        - 'cifar10' / 'mnist'           → 本地自定义模型（向后兼容）
        - 'resnet50' / 'vit_b_16' / ... → torchvision.models 中的任意模型
        - 'timm:model_name'             → timm 模型
        - 'timm:model_name.tag'         → 带特定预训练标签的 timm 模型

    Returns:
        (model, input_size)
    """
    print(f"\n[1/3] 加载模型: {model_spec}...")

    # --- 本地自定义模型 ---
    if model_spec == 'cifar10':
        model = CIFAR10Net()
        model_path = '../data/cifar_net.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device,
                                              weights_only=True))
            print(f"  CIFAR-10 模型已加载 (来自 {model_path})")
        else:
            print(f"  警告: {model_path} 未找到，使用随机初始化权重")
        input_size = (3, 32, 32)

    elif model_spec == 'mnist':
        model = MNISTNet()
        model_path = '../data/mnist_cnn.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device,
                                              weights_only=True))
            print(f"  MNIST 模型已加载 (来自 {model_path})")
        else:
            print(f"  警告: {model_path} 未找到，使用随机初始化权重")
        input_size = (1, 28, 28)

    # --- timm 模型 ---
    elif model_spec.startswith('timm:'):
        if not _HAS_TIMM:
            raise ImportError(
                "timm 未安装，无法加载 timm 模型。"
                "请运行: pip install timm")
        timm_name = model_spec[5:]  # 去掉 'timm:' 前缀
        try:
            model = timm.create_model(timm_name, pretrained=True)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  timm 模型 {timm_name} 已加载 ({n_params/1e6:.1f}M 参数)")
        except Exception as e:
            print(f"\n  [错误] 加载 timm 模型失败: {e}")
            print(f"  提示: 用以下命令查看可用模型:")
            print(f"    python -c \"import timm; print('\\n'.join(timm.list_models('*vit*')[:20]))\"")
            raise
        input_size = detect_input_size(model)

    # --- torchvision 模型 ---
    else:
        try:
            model = models.__dict__[model_spec](weights='DEFAULT')
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  {model_spec} (ImageNet 预训练, {n_params/1e6:.1f}M 参数) 已加载")
        except KeyError:
            print(f"\n  [错误] torchvision.models 中没有 '{model_spec}'")
            if _HAS_TIMM and 'vit' in model_spec.lower():
                print(f"  提示: ViT 模型请使用 'timm:{model_spec}'")
            print(f"  可用的 torchvision 分类模型:")
            print(f"    resnet50, resnet101, resnet152, vgg16, vgg19,")
            print(f"    mobilenet_v2, mobilenet_v3_large, densenet121, densenet201,")
            print(f"    efficientnet_b0-b7, regnet_y_400mf, regnet_y_32gf,")
            print(f"    convnext_tiny/base/large, swin_t/s/b,")
            print(f"    vit_b_16, vit_b_32, vit_l_16, maxvit_t,")
            print(f"    wide_resnet50_2, wide_resnet101_2,")
            print(f"    resnext50_32x4d, resnext101_64x4d")
            raise ValueError(f"不支持的模型: {model_spec}")
        input_size = detect_input_size(model)
        # inception 特殊处理
        if 'inception' in model_spec:
            input_size = (3, 299, 299)

    model = model.to(device)
    model.eval()
    return model, input_size


def load_data(dataset_name, batch_size=64):
    """
    加载测试数据集

    Returns:
        test_loader, classes
    """
    print(f"\n[2/3] 加载测试数据: {dataset_name}...")

    data_dir = '../data'

    if dataset_name == 'cifar10':
        # 注意: 与 torchtest.py 训练时的归一化保持一致 (mean=0.5, std=0.5)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        testset = datasets.CIFAR10(root=data_dir, train=False,
                                   download=True, transform=transform)
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    elif dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        testset = datasets.MNIST(root=data_dir, train=False,
                                 download=True, transform=transform)
        classes = tuple(str(i) for i in range(10))

    elif dataset_name == 'imagenet_val':
        # 使用本地 ImageNet 验证集子集（transferattack/data 文件夹）
        sys_path_ok = _import_transferattack_utils()
        if not sys_path_ok:
            raise ImportError(
                "无法导入 transferattack.utils (AdvDataset)，"
                "请确认 D:/py/transferattack 目录存在")

        from transferattack.utils import AdvDataset
        # 从 input_dir/images 读取 50 张原始测试图（eval=False）
        testset = AdvDataset(input_dir='../transferattack/data',
                             eval=False)
        classes = None
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=0)
        class_names = _load_imagenet_labels()
        return test_loader, class_names

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  测试样本数: {len(testset)}")
    return test_loader, classes


def _load_imagenet_labels():
    """加载 ImageNet 标签"""
    try:
        import requests
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        resp = requests.get(url, timeout=5)
        return [l.strip() for l in resp.text.splitlines()]
    except:
        return [f'class_{i}' for i in range(1000)]


def _import_transferattack_utils():
    """
    将 D:/py 加入 sys.path 以便导入 transferattack 包

    Returns:
        bool: 是否导入成功
    """
    import sys
    try:
        from transferattack import utils  # noqa: F401
        return True
    except ImportError:
        py_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if py_root not in sys.path:
            sys.path.insert(0, py_root)
        try:
            from transferattack import utils  # noqa: F401
            return True
        except ImportError:
            return False


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(
        description='GD-UAP: Generalizable Data-free Universal Adversarial Perturbations')
    parser.add_argument('--model', type=str, default='cifar10',
                        help='目标模型. 支持: cifar10, mnist, '
                             '任意 torchvision 模型 (resnet50, vit_b_16, '
                             'convnext_tiny, swin_t, ...), '
                             '或 timm 模型 (timm:vit_base_patch16_224)')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'mnist', 'imagenet_val'],
                        help='测试数据集 (默认: cifar10)')
    parser.add_argument('--eps', type=float, default=10/255,
                        help='L∞ 扰动预算 (默认: 10/255 ≈ 0.039)')
    parser.add_argument('--max-iter', type=int, default=2000,
                        help='最大迭代次数 (默认: 2000, 大模型建议 5000)')
    parser.add_argument('--lr', type=float, default=0.5,
                        help='学习率 (默认: 0.5, 大模型建议 0.1)')
    parser.add_argument('--noise-batch', type=int, default=1,
                        help='每轮迭代的随机噪声样本数 (默认: 1, 大模型建议 8)')
    parser.add_argument('--prior', type=str, default='gaussian',
                        choices=['black', 'range', 'gaussian', 'jigsaw'],
                        help='人工图像样本类型 (默认: gaussian). '
                             'black=全黑, range=均匀噪声, '
                             'gaussian=高斯噪声, jigsaw=拼图图像')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='评估时的批大小 (默认: 64)')
    parser.add_argument('--eval-only', action='store_true',
                        help='仅评估已保存的扰动，跳过生成阶段')
    parser.add_argument('--perturbation-path', type=str,
                        default='../data/gd_uap_perturbation.pt',
                        help='扰动文件的保存/加载路径')
    parser.add_argument('--single-image', type=str, default=None,
                        help='对单张图片进行评估 (指定图片路径)')
    parser.add_argument('--output-dir', type=str, default='../data',
                        help='输出目录 (默认: ../data)')
    args = parser.parse_args()

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  GD-UAP: 通用数据无关对抗扰动")
    print(f"  设备: {device}")
    print(f"  模型: {args.model}")
    print("=" * 60)

    # 加载模型
    model, input_size = load_model_flexible(args.model, device)

    # 大模型推荐参数提示
    if input_size[1] >= 224:
        print(f"\n  [Info] 检测到大模型 ({input_size[1]}×{input_size[2]})，"
              f"建议参数:")
        print(f"         --lr 0.1 --max-iter 5000 --noise-batch 8")

    # 初始化 GD-UAP
    gduap = GDUAP(
        model,
        eps=args.eps,
        max_iter=args.max_iter,
        lr=args.lr,
        input_size=input_size,
        noise_batch=args.noise_batch,
        prior=args.prior,
        device=device,
    )

    # 生成或加载扰动
    pert_path = os.path.join(args.output_dir,
                             os.path.basename(args.perturbation_path))
    if not pert_path.endswith('.pt'):
        pert_path += '.pt'

    if args.eval_only:
        v = gduap.load(pert_path)
    else:
        print(f"\n[3/3] 开始生成 GD-UAP 扰动...")
        v = gduap.generate(verbose=True)
        # 保存扰动
        os.makedirs(args.output_dir, exist_ok=True)
        gduap.save(v, pert_path)

    # ==================== 评估 ====================

    # 1. 单张图片评估
    if args.single_image:
        print(f"\n单张图片评估: {args.single_image}")
        test_img = args.single_image
        if not os.path.exists(test_img):
            # 尝试默认测试图片
            test_img = '../data/test_dog.jpg'
        # 小模型用类别索引，大模型用 ImageNet 标签
        if args.model in ('cifar10', 'mnist'):
            class_names = None
        else:
            class_names = _load_imagenet_labels()
        # CIFAR-10 训练时用了 Normalize(0.5,0.5)，单图评估需保持一致
        if args.model == 'cifar10':
            norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        else:
            norm_mean, norm_std = None, None
        evaluate_uap_single_image(model, v, test_img, device,
                                  class_names=class_names,
                                  norm_mean=norm_mean, norm_std=norm_std)
    else:
        # 尝试默认图片
        default_img = '../data/test_dog.jpg'
        if os.path.exists(default_img):
            print("\n对默认测试图片进行评估...")
            if args.model in ('cifar10', 'mnist'):
                class_names = None
            else:
                class_names = _load_imagenet_labels()
            if args.model == 'cifar10':
                norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            else:
                norm_mean, norm_std = None, None
            evaluate_uap_single_image(model, v, default_img, device,
                                      class_names=class_names,
                                      norm_mean=norm_mean, norm_std=norm_std)

    # 2. 数据集批量评估
    if args.dataset != 'imagenet_val':
        test_loader, classes = load_data(args.dataset, args.batch_size)
    else:
        test_loader, classes = load_data(args.dataset, args.batch_size)

    evaluate_uap(model, v, test_loader, device, verbose=True)

    # 3. 可视化训练 loss 曲线
    if hasattr(gduap, 'loss_history') and gduap.loss_history:
        plt.figure(figsize=(8, 4))
        plt.plot(gduap.loss_history, linewidth=1)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('GD-UAP Training Loss Curve')
        plt.grid(True, alpha=0.3)
        loss_plot_path = os.path.join(args.output_dir, 'gd_uap_loss.png')
        plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\nLoss 曲线已保存到: {loss_plot_path}")

    print("\n✅ GD-UAP 完成！")


if __name__ == '__main__':
    main()
