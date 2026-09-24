"""
对抗训练 (Adversarial Training) 实验
====================================
标准对抗训练 (Madry et al. 2018): 从随机初始化训练, 每 batch 实时用 PGD 生成
对抗样本加入训练, 让模型学会对对抗扰动鲁棒.

  * 训练: PGD-7, eps=8/255, alpha=2/255  (Madry 原版配方)
  * 从随机初始化开始 (注: 实验发现从已收敛 baseline 微调会因对抗损失过大而崩溃,
    标准做法是从头训练, 见报告说明)
  * 评估: AutoAttack (最严格, 与防御实验同标准) + PGD-20 白盒

用法:
  python adv_train.py              # 完整对抗训练 (30 epochs + AA 1000 样本)
  python adv_train.py --quick      # 冒烟 (2 epochs + AA 50 样本)
  python adv_train.py --reuse      # 复用已保存的 adv_train.pth
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

# 复用 try.py 中的模型定义 / 数据加载 / 评估 / 攻击函数
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
T = importlib.import_module("try")  # try 是保留字, 不能 import try

DATA_ROOT = T.DATA_ROOT
CKPT_DIR = T.CKPT_DIR
REPORT_DIR = T.REPORT_DIR
ROOT = T.ROOT


class C1Net(nn.Module):
    """加大版 LeNet: 4 卷积层 + 3 全连接, ~270K 参数, 适合 CIFAR-10 对抗训练.
    无 BatchNorm/残差, 保持与原 LeNet 同风格, CPU 友好."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def pgd_linf(model, x, y, eps, alpha, iters, device="cpu", rand_start=True):
    """标准 Linf PGD, 返回对抗样本 (Madry 训练时用的)."""
    model.eval()
    x0 = x.clone().detach()
    delta = torch.zeros_like(x0)
    if rand_start:
        delta.uniform_(-eps, eps)
        delta = torch.clamp(delta, -x0, 1 - x0)
    for _ in range(iters):
        delta = delta.detach().requires_grad_(True)
        x_adv = torch.clamp(x0 + delta, 0, 1)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, delta)[0]
        delta = delta.detach() + alpha * grad.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(delta, -x0, 1 - x0)
    return torch.clamp(x0 + delta, 0, 1)


def adv_train(model, train_loader, eval_loader, device, epochs, lr,
              eps, alpha, pgd_iters, progress=None, tag="adv"):
    """对抗训练: 每 batch 先 PGD 生成对抗样本, 再在对抗样本上算 CE 损失. (Madry 2018)"""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = n = 0
        t_ep = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            # 1) 生成对抗样本 (在训练用随机增强后的 x 上做 PGD)
            x_adv = pgd_linf(model, x, y, eps, alpha, pgd_iters, device)
            # 2) 在对抗样本上训练
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        acc = T.evaluate(model, eval_loader, device, desc=f"{tag} ep{ep}")
        if progress is not None:
            progress.set(current_model=tag, current_epoch=ep,
                         total_epochs=epochs, current_loss=run_loss / n,
                         current_acc=acc)
            progress.note_epoch_done(time.time() - t_ep)
            progress.set(eta_s=progress.estimate_eta())
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30, help="对抗训练 epochs")
    ap.add_argument("--arch", type=str, default="lenet",
                    choices=["lenet", "c1"],
                    help="lenet=原62K网络; c1=加大版~270K网络")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1, help="对抗训练学习率 (从头训练用较大)")
    ap.add_argument("--eps", type=float, default=8.0, help="Linf epsilon (1/255)")
    ap.add_argument("--pgd-iters", type=int, default=7, help="训练时 PGD 步数 (Madry 用 7)")
    ap.add_argument("--alpha", type=float, default=2.0, help="PGD 步长 (1/255), 默认 eps/4")
    ap.add_argument("--aa-samples", type=int, default=1000)
    ap.add_argument("--aa-batch", type=int, default=100)
    ap.add_argument("--aa-square-queries", type=int, default=1500)
    ap.add_argument("--aa-iters", type=int, default=100)
    ap.add_argument("--pgd-test-iters", type=int, default=20, help="测试 PGD 步数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="冒烟测试")
    ap.add_argument("--reuse", action="store_true", help="复用 adv_train.pth")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.aa_samples = 2, 50
        args.aa_square_queries = 200
        args.pgd_test_iters = 3

    device = "cpu"
    eps = args.eps / 255.0
    alpha = args.alpha / 255.0
    print(f"[env] device={device} epochs={args.epochs} eps={args.eps}/255 "
          f"pgd_train_iters={args.pgd_iters} aa_samples={args.aa_samples}")

    # 标准对抗训练: 从随机初始化训练 (Madry 2018)
    # 注: 实验发现从已收敛 baseline 微调会因对抗损失过大(约14x干净损失)导致 clean 崩到 10%,
    #     故采用标准做法从头训练.
    if args.arch == "c1":
        model = T.NormalizeNet(C1Net()).to(device)
        print(f"[init] 加大模型 C1 ({count_params(model)} 参数) 随机初始化")
    else:
        model = T.make_model("baseline", args.seed).to(device)
        print(f"[init] LeNet ({count_params(model)} 参数) 随机初始化 (Madry 标准做法)")

    train_loader, test_loader, testset = T.get_loaders(args.batch)

    # AA 与 PGD 共用同一批固定测试样本
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
    progress.set(phase="training", current_model="(对抗训练)")

    # ---- 对抗训练 ----
    adv_ckpt = os.path.join(CKPT_DIR, f"adv_train_{args.arch}.pth")
    if args.reuse and os.path.exists(adv_ckpt):
        model.load_state_dict(torch.load(adv_ckpt, map_location=device, weights_only=True))
        print(f"[load] adv_train <- {adv_ckpt}")
    else:
        print(f"[train] 对抗训练 (PGD-{args.pgd_iters}, eps={args.eps}/255) ...")
        progress.log(f"开始对抗训练 PGD-{args.pgd_iters}")
        t0 = time.time()
        adv_train(model, train_loader, test_loader, device, args.epochs, args.lr,
                  eps, alpha, args.pgd_iters, progress=progress)
        torch.save(model.state_dict(), adv_ckpt)
        print(f"[train] 对抗训练 done in {time.time()-t0:.0f}s -> {adv_ckpt}")
        progress.log(f"对抗训练完成 ({time.time()-t0:.0f}s)")

    results = {"seed": args.seed, "epochs": args.epochs, "eps": args.eps,
               "pgd_train_iters": args.pgd_iters, "aa_samples": args.aa_samples,
               "device": device, "models": {"adv_train": {}}}

    # ---- 干净准确率 ----
    print("\n========== 1) 干净测试准确率 ==========")
    progress.set(phase="eval_clean", current_model="(评估)")
    results["models"]["adv_train"]["clean"] = round(
        T.evaluate(model, test_loader, device, desc="adv_train"), 2)

    # ---- AutoAttack ----
    print("\n========== 2) AutoAttack (Linf, eps=8/255) ==========")
    progress.set(phase="attacking", current_attack="AutoAttack")
    rob, dt = T.run_autoattack(model, x_attack, y_attack, eps,
                               args.aa_batch, args.aa_square_queries,
                               args.aa_iters, device)
    results["models"]["adv_train"]["aa_robust"] = round(float(rob), 2)
    results["models"]["adv_train"]["aa_time_s"] = round(dt, 1)
    print(f"  [AA] adv_train robust acc = {rob:.2f}%  ({dt:.0f}s)")
    progress.set(models={"adv_train": results["models"]["adv_train"]})
    progress.log(f"AutoAttack 完成: 鲁棒率 {rob:.2f}%")

    # ---- PGD-20 白盒 (测试步数多于训练, 严格评估) ----
    print("\n========== 3) PGD-20 白盒攻击 ==========")
    progress.set(phase="attacking", current_attack=f"PGD-{args.pgd_test_iters}")
    acc = T.attack_eval(model, x_attack, y_attack, eps, alpha,
                        args.pgd_test_iters, device=device)
    results["models"]["adv_train"]["pgd20"] = round(acc, 2)
    print(f"  [PGD] adv_train robust acc = {acc:.2f}%")
    progress.set(models={"adv_train": results["models"]["adv_train"]})

    report_path = os.path.join(REPORT_DIR, f"adv_train_{args.arch}_experiment.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[report] 已保存 -> {report_path}")
    progress.set(status="done", phase="done", current_model="(完成)")
    progress.log("对抗训练实验完成")


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
