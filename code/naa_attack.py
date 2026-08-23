"""
NAA: Neuron Attribution-based Attack 实验脚本
=============================================

使用 transferattack 仓库中的 NAA 实现 (CVPR 2022, Neuron Attribution-Based Attack,
https://arxiv.org/pdf/2204.00008.pdf) 做迁移性攻击实验。

设计要点
--------
1. **CPU 补丁**: transferattack 的 NAA 原实现里硬编码了 ``.cuda()``,本环境是
   CPU-only (torch 2.11.0+cpu),因此派生 ``NAA_CPU`` 把设备调用改为 ``self.device``。
   hook 产生的中间张量 (``mid_grad`` / ``mid_output``) 通过模块命名空间访问。

2. **伪标签 (pseudo-label) 攻击**: ``transferattack/data/images`` 里的 50 张图其实是
   32×32 的 CIFAR-10 图像放大到 224×224,labels.csv 里是 CIFAR 类别 (0-9),对
   ImageNet 预训练模型没有意义。因此按无标签迁移攻击的标准做法,用代理模型自己
   的预测作为攻击目标类 (untargeted)。

3. **高级模型**: 本环境已缓存 resnet18/resnet50;其余从 torchvision 下载
   (download.pytorch.org 可直连): vgg16, densenet121, mobilenet_v2, inception_v3,
   convnext_tiny, swin_t, vit_b_16 —— 覆盖经典 CNN 与现代 Transformer/ConvNeXt。
   (注: timm 模型在本环境无法下载,因 huggingface_hub 1.14.0 有 httpx 客户端
   关闭的 bug,故不在默认列表内。)

4. **评估**:
   - 白盒 ASR: 代理模型对扰动后图像的预测是否改变
   - 黑盒迁移 ASR: 其他模型 (受害者) 对扰动后图像的预测是否改变

用法
----
    python naa_attack.py                                     # 默认: NAA(layer1) vs MIFGSM, 50 张图
    python naa_attack.py --images 20 --quick                 # 快速冒烟测试
    python naa_attack.py --layers layer1,layer2,layer3,layer4  # NAA 特征层扫描
    python naa_attack.py --surrogate resnet18 --targets convnext_tiny,swin_t
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# --- 环境修复 ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Windows + Anaconda 的 OpenMP 冲突
PY_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PY_ROOT not in sys.path:
    sys.path.insert(0, PY_ROOT)

from transferattack.advanced_objective import naa as naa_module
from transferattack.advanced_objective.naa import NAA
from transferattack.gradient.mifgsm import MIFGSM
from transferattack.utils import wrap_model

import torchvision.models as models

# 高级模型池 (torchvision, 已验证可加载)
ADVANCED_MODELS = [
    "resnet18", "resnet50", "vgg16", "densenet121", "mobilenet_v2",
    "inception_v3", "convnext_tiny", "swin_t", "vit_b_16",
]

IMAGE_DIR = os.path.join(PY_ROOT, "transferattack", "data", "images")
PICTURE_DIR = os.path.join(PY_ROOT, "picture")


# ==================== CPU 兼容的 NAA ====================

class NAA_CPU(NAA):
    """
    将 transferattack 原版 NAA 的硬编码 ``.cuda()`` 改为 ``self.device``,
    使其可在 CPU/GPU 上运行。其余逻辑 (特征层 hook、梯度聚合、迭代更新) 保持不变。
    """

    def forward(self, data, label, **kwargs):
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        delta = self.init_delta(data)

        h = self.feature_layer.register_forward_hook(self._NAA__forward_hook)
        h2 = self.feature_layer.register_full_backward_hook(self._NAA__backward_hook)

        # 阶段 1: 沿从全零图像到真实图像的线性路径聚合特征层梯度,得到神经元归因
        agg_grad = 0
        for iter_n in range(self.N):
            x_m = torch.zeros(data.size(), device=self.device)
            x_m = x_m + data.clone().detach() * iter_n / self.N
            out = torch.softmax(self.model(x_m), 1)
            loss = 0
            for batch_i in range(data.shape[0]):
                loss += out[batch_i][label[batch_i]]
            self.model.zero_grad()
            loss.backward()
            agg_grad += naa_module.mid_grad[0].detach()
        agg_grad /= self.N
        h2.remove()

        # 阶段 2: 使特征图偏离其在零图像上的基准,并沿归因方向更新
        x_prime = torch.zeros(data.size(), device=self.device)
        self.model(x_prime)
        y_prime = naa_module.mid_output.detach().clone()
        for _ in range(self.epoch):
            logits = self.get_logits(self.transform(data + delta))
            loss = ((naa_module.mid_output - y_prime) * agg_grad).sum()
            self.model.zero_grad()
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            delta = self.update_delta(delta, data, -grad, self.alpha)
        h.remove()
        return delta.detach()


# ==================== 工具函数 ====================

def load_wrapped_model(name, device):
    """加载 torchvision 模型并套上预处理包装 (与 transferattack.Attack 一致)。"""
    print(f"  ... 加载 {name} ...")
    m = models.__dict__[name](weights="DEFAULT")
    model = wrap_model(m.eval()).to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_images(n=None):
    """加载全部/前 n 张测试图像 → (n,3,224,224) float32 [0,1] + 文件名列表。"""
    files = sorted(f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".png"))
    if n is not None:
        files = files[:n]
    imgs, names = [], []
    for f in files:
        im = Image.open(os.path.join(IMAGE_DIR, f)).convert("RGB").resize((224, 224))
        imgs.append(np.array(im).astype(np.float32) / 255.0)
        names.append(f)
    data = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).contiguous()
    return data, names


def pseudo_labels(model, data, device):
    """用代理模型自身预测作为攻击目标 (无标签迁移攻击标准做法)。"""
    with torch.no_grad():
        return model(data.to(device)).argmax(dim=1).cpu()


def run_attack(attacker, data, labels, batch_size, device):
    """分批执行攻击,拼接所有扰动后的对抗样本。"""
    adv_parts = []
    for i in range(0, data.shape[0], batch_size):
        x = data[i:i + batch_size].to(device)
        y = labels[i:i + batch_size].to(device)
        delta = attacker(x, y)
        adv_parts.append((x + delta).clamp(0, 1).cpu())
    return torch.cat(adv_parts, 0)


def white_box_asr(model, clean, adv, device):
    """白盒 ASR: 代理模型预测是否改变。"""
    with torch.no_grad():
        pc = model(clean.to(device)).argmax(1).cpu()
        pa = model(adv.to(device)).argmax(1).cpu()
    return (pc != pa).float().mean().item() * 100


def transfer_asr(model, clean, adv, device):
    """黑盒迁移 ASR: 受害者模型预测是否改变。"""
    return white_box_asr(model, clean, adv, device)


def save_adv_images(adv, names, tag):
    """保存对抗样本到 picture/naa_adv/ 便于目检。"""
    out_dir = os.path.join(PICTURE_DIR, "naa_adv", tag)
    os.makedirs(out_dir, exist_ok=True)
    for img, name in zip(adv, names):
        arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(out_dir, name))


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(description="NAA 迁移性攻击实验")
    parser.add_argument("--surrogate", type=str, default="resnet50",
                        help="代理模型 (白盒攻击对象, 默认 resnet50)")
    parser.add_argument("--targets", type=str, default=",".join(ADVANCED_MODELS),
                        help="受害者模型列表 (黑盒迁移评估)")
    parser.add_argument("--layers", type=str, default="layer1",
                        help="NAA 特征层, 逗号分隔可扫描 (如 layer1,layer2,layer3,layer4)")
    parser.add_argument("--eps", type=float, default=16 / 255)
    parser.add_argument("--alpha", type=float, default=1.6 / 255)
    parser.add_argument("--epoch", type=int, default=10, help="NAA/MI-FGSM 迭代次数")
    parser.add_argument("--N", type=int, default=30, help="NAA 梯度聚合的插值点数")
    parser.add_argument("--decay", type=float, default=1.0, help="动量衰减")
    parser.add_argument("--batch-size", type=int, default=10, help="攻击批大小 (CPU 建议 10)")
    parser.add_argument("--images", type=int, default=None, help="使用的图像数 (默认全部 50)")
    parser.add_argument("--quick", action="store_true", help="快速模式: epoch=3, N=6")
    parser.add_argument("--no-mifgsm", action="store_true", help="不运行 MIFGSM 基线")
    parser.add_argument("--save-images", action="store_true", help="保存对抗样本图像")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("  NAA: Neuron Attribution-based Attack — transferattack 复现实验")
    print(f"  设备: {device} | 代理模型: {args.surrogate} | 图像: {args.images or 50} 张")
    print(f"  eps={args.eps:.4f} alpha={args.alpha:.4f} epoch={args.epoch} N={args.N}")
    print("=" * 78)

    if args.quick:
        args.epoch, args.N = 3, 6
        print("  [quick] epoch=3, N=6")

    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    layers = [l.strip() for l in args.layers.split(",") if l.strip()]

    # ---------- 加载数据 ----------
    data, names = load_images(args.images)
    print(f"\n[数据] 加载 {data.shape[0]} 张图: {IMAGE_DIR}")
    if args.save_images:
        save_adv_images(data, names, "clean")

    # ---------- 加载代理模型 ----------
    print(f"\n[代理模型] {args.surrogate}")
    surrogate = load_wrapped_model(args.surrogate, device)
    labels = pseudo_labels(surrogate, data, device)

    configs = [(f"NAA-{l}", l) for l in layers]
    if not args.no_mifgsm:
        configs.append(("MIFGSM", None))

    results = []  # 每行: attack, target, asr
    adv_store = {}

    for attack_name, layer in configs:
        print(f"\n{'─' * 78}\n[攻击] {attack_name}"
              + (f" (特征层 {layer})" if layer else " (MI-FGSM 基线)")
              + "\n" + "─" * 78)

        t0 = time.time()
        if layer is not None:
            attacker = NAA_CPU(model_name=args.surrogate, epsilon=args.eps,
                               alpha=args.alpha, epoch=args.epoch, decay=args.decay,
                               num_ens=args.N, N=args.N, feature_layer=layer,
                               device=device)
        else:
            attacker = MIFGSM(model_name=args.surrogate, epsilon=args.eps,
                              alpha=args.alpha, epoch=args.epoch, decay=args.decay,
                              device=device)

        adv = run_attack(attacker, data, labels, args.batch_size, device)
        print(f"  [攻击完成] 耗时 {time.time() - t0:.1f}s")

        # 白盒评估
        asr = white_box_asr(surrogate, data, adv, device)
        wb_name = f"{args.surrogate}(白盒)"
        print(f"  {wb_name:24s} {asr:5.1f}%")
        results.append({"attack": attack_name, "target": wb_name, "asr": asr})
        adv_store[attack_name] = adv

        # 黑盒迁移评估
        for tname in targets:
            if tname == args.surrogate:
                continue
            tm = load_wrapped_model(tname, device)
            tasr = transfer_asr(tm, data, adv, device)
            print(f"  {'黑盒迁移 ' + tname:24s} {tasr:5.1f}%")
            results.append({"attack": attack_name, "target": tname, "asr": tasr})
            del tm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.save_images:
            save_adv_images(adv, names, attack_name)

    # ---------- 汇总 ----------
    print("\n" + "=" * 78)
    print("  结果汇总: 攻击成功率 ASR (%)")
    print("=" * 78)
    eval_targets = [t for t in targets if t != args.surrogate]
    header = ["attack"] + [f"{args.surrogate}(白盒)"] + eval_targets
    print("  " + " | ".join(f"{h:>22s}" for h in header))
    for attack_name, _ in configs:
        row = {r["target"]: r["asr"] for r in results if r["attack"] == attack_name}
        line = f"  {attack_name:>22s}"
        for h in header[1:]:
            line += f" | {row.get(h, float('nan')):>20.1f}"
        print(line)

    os.makedirs(PICTURE_DIR, exist_ok=True)
    csv_path = os.path.join(PICTURE_DIR, "naa_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["attack", "target", "asr"])
        w.writeheader()
        w.writerows(results)
    print(f"\n  结果已保存: {csv_path}")
    print("\n✅ NAA 实验完成")


if __name__ == "__main__":
    main()
