"""
输入扩散变换防御实验 (Diffusion Transform Defense)
==================================================
假设: 训练和推理都对输入施加"像素向周围八格扩散"的变换, 观察对抗扰动是否会
被平滑/吸收进扩散分布而"被免疫".

三种扩散模式 (训练与推理保持一致):
  fixed   : 每个像素固定扩散 1% 到 8 邻居
  spatial : 每个像素位置随机一个扩散比例 (空间随机)
  global  : 整张图每次随机一个扩散比例 (全局随机)

数学: 用 3x3 卷积核实现. 设每个像素的扩散比例 p, 则
  新值 = (1-p)*自身 + (p/8)*(8个邻居之和)
即把自身能量按比例 p 分给邻居, 总能量守恒.

评估: AutoAttack (Linf eps=8/255) + PGD-20 白盒, 与对抗训练(26.8%)对比.

用法:
  python diffuse_train.py --mode fixed
  python diffuse_train.py --mode spatial
  python diffuse_train.py --mode global
  python diffuse_train.py --quick --mode spatial
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
T = importlib.import_module("try")  # try 是保留字
import adv_train as A

DATA_ROOT = T.DATA_ROOT
CKPT_DIR = T.CKPT_DIR
REPORT_DIR = T.REPORT_DIR
ROOT = T.ROOT


class DiffuseNet(nn.Module):
    """对输入做"像素向周围八格扩散"变换后再喂给 C1 网络.
    训练与推理用同一变换 (mode 决定)."""
    def __init__(self, net, mode="fixed", p=0.01):
        super().__init__()
        self.net = net
        self.mode = mode
        self.p = p  # 固定模式的扩散比例 (或随机比例的上限)
        # 3x3 均值卷积核 (按通道 group=3, 每像素的 8 个邻居在同通道内)
        kernel = torch.ones(3, 1, 3, 3) / 8.0
        kernel[:, 0, 1, 1] = 0.0  # 去掉中心, 只留 8 邻居
        self.register_buffer("kernel", kernel)

    def _neighbor_avg(self, x):
        return F.conv2d(x, self.kernel, padding=1, groups=3)  # 8 邻居均值

    def _diffuse_fixed(self, x):
        # 每个像素扩散比例固定 = self.p
        nb = self._neighbor_avg(x)
        return (1 - self.p) * x + self.p * nb

    def _diffuse_spatial(self, x):
        # 每个像素位置随机一个扩散比例 (0 ~ self.p), 空间随机
        p_map = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3],
                           device=x.device) * self.p
        nb = self._neighbor_avg(x)
        return (1 - p_map) * x + p_map * nb

    def _diffuse_global(self, x):
        # 整张图每次随机一个扩散比例 (0 ~ self.p), 全局随机
        p_val = torch.rand(x.shape[0], 1, 1, 1, device=x.device) * self.p
        nb = self._neighbor_avg(x)
        return (1 - p_val) * x + p_val * nb

    def forward(self, x):
        # 先归一化到 [0,1]? 不行, 归一化应该由内部 net 处理.
        # 注意: C1Net 期望输入已在 [0,1], 且 DiffuseNet 内部要归一化.
        if self.mode == "fixed":
            xd = self._diffuse_fixed(x)
        elif self.mode == "spatial":
            xd = self._diffuse_spatial(x)
        elif self.mode == "global":
            xd = self._diffuse_global(x)
        else:
            xd = x
        return self.net((xd - 0.5) / 0.5)  # 归一化到 [-1,1] 喂给 C1


def train_clean(model, train_loader, eval_loader, device, epochs, lr, progress=None):
    """普通 (非对抗) 训练, 输入在 forward 里已做扩散变换."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = n = 0
        t_ep = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        acc = T.evaluate(model, eval_loader, device, desc=f"diffuse ep{ep}")
        if progress is not None:
            progress.set(current_model="diffuse", current_epoch=ep,
                         total_epochs=epochs, current_loss=run_loss / n,
                         current_acc=acc)
            progress.note_epoch_done(time.time() - t_ep)
            progress.set(eta_s=progress.estimate_eta())
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="fixed",
                    choices=["fixed", "spatial", "global", "none"])
    ap.add_argument("--p", type=float, default=0.01, help="扩散比例上限 (fixed 用固定值)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--eps", type=float, default=8.0)
    ap.add_argument("--aa-samples", type=int, default=1000)
    ap.add_argument("--aa-batch", type=int, default=100)
    ap.add_argument("--aa-square-queries", type=int, default=1500)
    ap.add_argument("--aa-iters", type=int, default=100)
    ap.add_argument("--pgd-test-iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.aa_samples = 2, 50
        args.aa_square_queries = 200
        args.pgd_test_iters = 3

    device = "cpu"
    eps = args.eps / 255.0
    alpha = 2.0 / 255.0
    print(f"[env] mode={args.mode} p={args.p} epochs={args.epochs} eps={args.eps}/255")

    # 模型: C1 + 扩散变换
    model = DiffuseNet(A.C1Net(), mode=args.mode, p=args.p).to(device)
    print(f"[init] DiffuseNet({args.mode}) + C1Net ({A.count_params(model)} 参数)")

    train_loader, test_loader, testset = T.get_loaders(args.batch)

    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(testset), size=args.aa_samples, replace=False)
    sub = Subset(testset, idx)
    sub_loader = DataLoader(sub, batch_size=args.aa_batch, shuffle=False, num_workers=0)
    xs, ys = [], []
    for x, y in sub_loader:
        xs.append(x); ys.append(y)
    x_attack = torch.cat(xs).to(device)
    y_attack = torch.cat(ys).to(device)

    progress = T.Progress(os.path.join(REPORT_DIR, "progress.json"))
    progress.set(phase="training", current_model=f"(diffuse-{args.mode})")

    ckpt = os.path.join(CKPT_DIR, f"diffuse_{args.mode}.pth")
    print(f"[train] 扩散变换训练 (mode={args.mode}) ...")
    t0 = time.time()
    train_clean(model, train_loader, test_loader, device, args.epochs, args.lr, progress)
    torch.save(model.state_dict(), ckpt)
    print(f"[train] done in {time.time()-t0:.0f}s -> {ckpt}")

    results = {"mode": args.mode, "p": args.p, "epochs": args.epochs,
               "eps": args.eps, "aa_samples": args.aa_samples,
               "device": device, "models": {"diffuse": {}}}

    # 干净准确率
    print("\n========== 1) 干净测试准确率 ==========")
    progress.set(phase="eval_clean", current_model="(评估)")
    results["models"]["diffuse"]["clean"] = round(
        T.evaluate(model, test_loader, device, desc="diffuse"), 2)

    # AutoAttack
    print("\n========== 2) AutoAttack (Linf, eps=8/255) ==========")
    progress.set(phase="attacking", current_attack="AutoAttack")
    rob, dt = T.run_autoattack(model, x_attack, y_attack, eps,
                               args.aa_batch, args.aa_square_queries,
                               args.aa_iters, device)
    results["models"]["diffuse"]["aa_robust"] = round(float(rob), 2)
    results["models"]["diffuse"]["aa_time_s"] = round(dt, 1)
    print(f"  [AA] diffuse-{args.mode} robust acc = {rob:.2f}%  ({dt:.0f}s)")
    progress.set(models={"diffuse": results["models"]["diffuse"]})
    progress.log(f"AutoAttack 完成: 鲁棒率 {rob:.2f}%")

    # PGD-20 白盒
    print("\n========== 3) PGD-20 白盒攻击 ==========")
    progress.set(phase="attacking", current_attack=f"PGD-{args.pgd_test_iters}")
    acc = T.attack_eval(model, x_attack, y_attack, eps, alpha,
                        args.pgd_test_iters, device=device)
    results["models"]["diffuse"]["pgd20"] = round(acc, 2)
    print(f"  [PGD] diffuse-{args.mode} robust acc = {acc:.2f}%")
    progress.set(models={"diffuse": results["models"]["diffuse"]})

    report_path = os.path.join(REPORT_DIR, f"diffuse_{args.mode}_experiment.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[report] 已保存 -> {report_path}")
    progress.set(status="done", phase="done", current_model="(完成)")
    progress.log("扩散变换实验完成")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            pp = os.path.join(REPORT_DIR, "progress.json")
            with open(pp, encoding="utf-8") as f:
                data = json.load(f)
            data["status"] = "error"
            data["error"] = traceback.format_exc()[-1500:]
            with open(pp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass
        raise
