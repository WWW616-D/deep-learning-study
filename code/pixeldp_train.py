"""
PixelDP 实验: 基于差分隐私的认证鲁棒性
=====================================
论文: Certified Robustness to Adversarial Examples with Differential Privacy
      (Lecuyer et al., IEEE S&P 2019)

核心思想: 把输入图像的每个像素看作数据库中的一条"记录", 在输入端加入 DP 噪声
(Gaussian/Laplace 机制), 使整个网络成为 (eps, delta)-PixelDP. 由 DP 的后处理性质,
该保证传递到输出; 再用 Proposition 1 的稳定性界, 可对每个样本给出**可认证的
鲁棒半径 L_max**(无论攻击如何实现, 只要扰动 p-范数 <= L_max 就无法改变预测).

  * 噪声层: 加在输入像素 (g=identity, 敏感度 Delta=1, 2-norm)
  * Gaussian 机制: sigma = sqrt(2 ln(1.25/delta)) * L / eps,  取 (eps=1.0, delta=0.05)
  * 训练: 每次前向都加噪 (论文做法), 普通 CE 损失
  * 预测: 多次加噪取蒙特卡洛均值估计 E(A(x)), 用 Hoeffding 界做置信区间,
          再用 Proposition 2 (广义鲁棒条件) 解出每个样本的 L_max
  * 评估: 干净准确率 / 认证准确率-半径曲线 / EOT-PGD / AutoAttack(rand, EOT)

用法:
  python pixeldp_train.py --L 0.1                 # 完整 (L=构造攻击界)
  python pixeldp_train.py --L 0.1 --quick         # 冒烟
"""

import argparse
import json
import math
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
T = importlib.import_module("try")
import adv_train as A

CKPT_DIR = T.CKPT_DIR
REPORT_DIR = T.REPORT_DIR
ROOT = T.ROOT


# ---------------- PixelDP 噪声层 ----------------
class PixelDPNet(nn.Module):
    """在输入像素上加 DP 噪声 (g=identity, Delta=1), 再送入基础网络.

    sigma 由 (eps, delta, L) 决定:
      Gaussian: sigma = sqrt(2 ln(1.25/delta)) * L / eps   (给出 (eps,delta)-DP, eps<=1)
      Laplace : sigma = sqrt(2) * L / eps                  (给出 (eps,0)-DP)
    """
    def __init__(self, base, sigma, mech="gaussian"):
        super().__init__()
        self.base = base          # 期望输入 [0,1]
        self.sigma = sigma
        self.mech = mech

    def sample_noise(self, x):
        if self.mech == "gaussian":
            return torch.randn_like(x) * self.sigma
        else:  # laplace: 标准差 = sqrt(2)*b, b = sigma/sqrt(2) => 用 std 参数化
            return torch.distributions.Laplace(0.0, self.sigma / math.sqrt(2)).sample(x.shape).to(x.device)

    def forward(self, x):
        # 训练与预测都加噪 (论文: 每次调用网络都加噪)
        xn = torch.clamp(x + self.sample_noise(x), 0.0, 1.0)
        return self.base(xn)


def sigma_from_L(L, eps, delta, mech="gaussian"):
    if mech == "gaussian":
        return math.sqrt(2 * math.log(1.25 / delta)) * L / eps
    return math.sqrt(2) * L / eps


# ---------------- 训练 ----------------
def train_pixeldp(model, train_loader, eval_loader, device, epochs, lr, progress=None):
    """论文做法: 用原损失和优化器训练, 每次前向都加 DP 噪声."""
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
        acc = T.evaluate(model, eval_loader, device, desc=f"pixeldp ep{ep}")
        if progress is not None:
            progress.set(current_model="pixeldp", current_epoch=ep,
                         total_epochs=epochs, current_loss=run_loss / n,
                         current_acc=acc)
            progress.note_epoch_done(time.time() - t_ep)
            progress.set(eta_s=progress.estimate_eta())
    return model


# ---------------- 认证 (Proposition 2) ----------------
@torch.no_grad()
def monte_carlo_scores(model, x, n_draws, device, group=100):
    """估计 E(A(x)): 每个样本重复 n_draws 次前向取均值. 返回 [N,K] 均值."""
    N = x.size(0)
    acc = None
    done = 0
    model.eval()
    while done < n_draws:
        g = min(group, n_draws - done)
        xr = x.repeat(g, 1, 1, 1)          # [g*N,3,H,W]
        p = F.softmax(model(xr), dim=1).reshape(g, N, -1).sum(0)
        acc = p if acc is None else acc + p
        done += g
    return acc / n_draws


def certified_radius(lb_top, ub_second, sigma, eps, delta, mech="gaussian",
                     L_hi=5.0, iters=60):
    """解出满足 Proposition 2 的最大 L.
    lb_top > exp(2*eps(L)) * ub_second + (1+exp(eps(L)))*delta,  其中 eps(L)=c*L/sigma
    (sigma 固定, 由构造界 L 与 eps 决定 => 认证时必须用 eps(L)=c*L/sigma 反过来解 L)
    """
    if mech == "gaussian":
        c = math.sqrt(2 * math.log(1.25 / delta))
    else:
        c = math.sqrt(2)

    def cond(L):
        epsL = c * L / sigma
        return lb_top > math.exp(2 * epsL) * ub_second + (1 + math.exp(epsL)) * delta

    if not cond(0.0):
        return 0.0
    if cond(L_hi):
        return L_hi
    lo, hi = 0.0, L_hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if cond(mid):
            lo = mid
        else:
            hi = mid
    return lo


@torch.no_grad()
def certify_dataset(model, x, y, sigma, delta, n_draws, alpha, device, bs=50):
    """返回每样本: 预测是否正确, L_max 认证半径, 以及干净概率向量."""
    model.eval()
    K = 10
    hoeff = math.sqrt(math.log(2 * K / (1 - alpha)) / (2 * n_draws))
    correct, lmax_list = [], []
    N = x.size(0)
    for i in range(0, N, bs):
        xb = x[i:i + bs]
        yb = y[i:i + bs]
        mean_p = monte_carlo_scores(model, xb, n_draws, device)
        lb = torch.clamp(mean_p - hoeff, 0, 1)
        ub = torch.clamp(mean_p + hoeff, 0, 1)
        pred = mean_p.argmax(1)
        for j in range(xb.size(0)):
            p = mean_p[j]
            top = pred[j].item()
            second = torch.cat([ub[j][:top], ub[j][top + 1:]]).max().item()
            L = certified_radius(lb[j][top].item(), second, sigma, 1.0, delta)
            correct.append(int(pred[j].item() == yb[j].item()))
            lmax_list.append(L)
    return np.array(correct), np.array(lmax_list), None


# ---------------- EOT 攻击 ----------------
def eot_pgd(model, x, y, eps, alpha, iters, n_eot, device):
    """EOT-PGD: 每步梯度对 n_eot 次噪声抽样取平均 (论文对随机化防御的适配做法)."""
    model.eval()
    x0 = x.clone().detach()
    delta = torch.zeros_like(x0).uniform_(-eps, eps).clamp(-x0, 1 - x0)
    for _ in range(iters):
        delta = delta.detach().requires_grad_(True)
        grad = torch.zeros_like(delta)
        for _ in range(n_eot):
            xa = torch.clamp(x0 + delta, 0, 1)
            loss = F.cross_entropy(model(xa), y)
            grad += torch.autograd.grad(loss, delta)[0]
        delta = (delta.detach() + alpha * (grad / n_eot).sign()).clamp(-eps, eps).clamp(-x0, 1 - x0)
    x_adv = torch.clamp(x0 + delta, 0, 1)
    with torch.no_grad():
        # 评估时也对噪声取平均 (蒙特卡洛预测)
        accs = []
        for _ in range(5):
            accs.append((model(x_adv).argmax(1) == y).float().mean().item())
    return float(np.mean(accs)) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=0.1, help="构造攻击界 (决定噪声大小)")
    ap.add_argument("--eps", type=float, default=1.0, help="DP 参数 eps (论文用 1.0)")
    ap.add_argument("--delta", type=float, default=0.05, help="DP 参数 delta (论文用 0.05)")
    ap.add_argument("--mech", type=str, default="gaussian", choices=["gaussian", "laplace"])
    ap.add_argument("--arch", type=str, default="c1", choices=["c1", "lenet"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--cert-samples", type=int, default=1000, help="认证用的测试样本数")
    ap.add_argument("--draws", type=int, default=500, help="蒙特卡洛噪声抽样次数")
    ap.add_argument("--alpha", type=float, default=0.001, help="认证置信度 1-alpha")
    ap.add_argument("--aa-samples", type=int, default=500)
    ap.add_argument("--aa-batch", type=int, default=100)
    ap.add_argument("--eps-attack", type=float, default=8.0, help="Linf 攻击预算 (1/255)")
    ap.add_argument("--pgd-iters", type=int, default=20)
    ap.add_argument("--eot", type=int, default=10, help="EOT 抽样次数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-aa", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.cert_samples, args.draws = 2, 100, 50
        args.aa_samples, args.pgd_iters, args.eot = 50, 5, 3

    device = "cpu"
    sigma = sigma_from_L(args.L, args.eps, args.delta, args.mech)
    print(f"[env] device={device} arch={args.arch} L={args.L} eps={args.eps} "
          f"delta={args.delta} mech={args.mech} -> sigma={sigma:.4f}")
    print(f"[env] epochs={args.epochs} draws={args.draws} cert_samples={args.cert_samples}")

    base = T.NormalizeNet(A.C1Net()) if args.arch == "c1" else T.make_model("baseline", args.seed)
    model = PixelDPNet(base, sigma, args.mech).to(device)
    print(f"[init] PixelDP + {args.arch} ({A.count_params(model)} 参数), sigma={sigma:.4f}")

    train_loader, test_loader, testset = T.get_loaders(args.batch)

    progress = T.Progress(os.path.join(REPORT_DIR, "progress.json"))
    progress.set(phase="training", current_model=f"(pixeldp L={args.L})")

    tag = f"pixeldp_L{args.L}_{args.arch}"
    ckpt = os.path.join(CKPT_DIR, f"{tag}.pth")
    print(f"[train] PixelDP 训练 (加噪训练) ...")
    progress.log(f"开始 PixelDP 训练 L={args.L} sigma={sigma:.4f}")
    t0 = time.time()
    train_pixeldp(model, train_loader, test_loader, device, args.epochs, args.lr, progress)
    torch.save(model.state_dict(), ckpt)
    print(f"[train] done in {time.time()-t0:.0f}s -> {ckpt}")
    progress.log(f"训练完成 ({time.time()-t0:.0f}s)")

    results = {"L": args.L, "eps": args.eps, "delta": args.delta, "mech": args.mech,
               "sigma": round(sigma, 4), "arch": args.arch, "epochs": args.epochs,
               "draws": args.draws, "alpha": args.alpha, "device": device}

    # ---- 干净准确率 (蒙特卡洛平均预测) ----
    print("\n========== 1) 干净准确率 (含噪推断) ==========")
    progress.set(phase="eval_clean", current_model="(评估)")
    rng = np.random.RandomState(args.seed)
    cert_idx = rng.choice(len(testset), size=args.cert_samples, replace=False)
    x_cert = torch.stack([testset[i][0] for i in cert_idx]).to(device)
    y_cert = torch.tensor([testset[i][1] for i in cert_idx]).to(device)
    # 单次加噪的干净准确率 (与普通模型可比)
    accs = []
    for _ in range(10):
        with torch.no_grad():
            accs.append((model(x_cert).argmax(1) == y_cert).float().mean().item() * 100)
    clean_single = float(np.mean(accs))
    print(f"  [clean] 单次加噪推断准确率 = {clean_single:.2f}%")
    results["clean_single"] = round(clean_single, 2)

    # ---- 认证 ----
    print(f"\n========== 2) 认证 (MC draws={args.draws}, alpha={args.alpha}) ==========")
    progress.set(phase="attacking", current_attack="认证计算")
    t0 = time.time()
    correct, lmax, _ = certify_dataset(model, x_cert, y_cert, sigma, args.delta,
                                       args.draws, args.alpha, device)
    cert_time = time.time() - t0
    clean_mc = correct.mean() * 100
    print(f"  [cert] 蒙特卡洛预测准确率 = {clean_mc:.2f}%  ({cert_time:.0f}s)")
    print(f"  [cert] 平均认证半径 L_max = {lmax.mean():.4f}, 中位数 = {np.median(lmax):.4f}")

    thresholds = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    cert_acc = {}
    for Tt in thresholds:
        both = ((correct == 1) & (lmax >= Tt)).mean() * 100
        cert_acc[str(Tt)] = round(float(both), 2)
        print(f"  [cert] T={Tt:<4} 认证准确率 = {both:.2f}%")
    results["clean_mc"] = round(float(clean_mc), 2)
    results["cert_time_s"] = round(cert_time, 1)
    results["mean_lmax"] = round(float(lmax.mean()), 4)
    results["median_lmax"] = round(float(np.median(lmax)), 4)
    results["certified_accuracy"] = cert_acc
    # 认证覆盖率 (有多少样本拿到 >0 的半径)
    results["cert_rate_positive"] = round(float((lmax > 0).mean() * 100), 2)
    print(f"  [cert] 拿到正认证半径的样本比例 = {results['cert_rate_positive']:.2f}%")
    progress.set(models={tag: {"clean": results["clean_mc"],
                               "mean_lmax": results["mean_lmax"],
                               "cert@0.1": cert_acc.get("0.1")}})
    progress.log(f"认证完成: 平均半径 {results['mean_lmax']}, 认证准确率@0.1={cert_acc.get('0.1')}%")

    # ---- EOT-PGD 攻击 (ε=8/255 Linf) ----
    eps = args.eps_attack / 255.0
    alpha = 2.0 / 255.0
    print(f"\n========== 3) EOT-PGD-{args.pgd_iters} (Linf eps={args.eps_attack}/255, EOT={args.eot}) ==========")
    progress.set(phase="attacking", current_attack="EOT-PGD")
    aa_idx = rng.choice(len(testset), size=args.aa_samples, replace=False)
    x_aa = torch.stack([testset[i][0] for i in aa_idx]).to(device)
    y_aa = torch.tensor([testset[i][1] for i in aa_idx]).to(device)
    acc_eot = eot_pgd(model, x_aa, y_aa, eps, alpha, args.pgd_iters, args.eot, device)
    print(f"  [EOT-PGD] 鲁棒准确率 = {acc_eot:.2f}%")
    results["eot_pgd20"] = round(acc_eot, 2)
    progress.set(models={tag: {**progress.data["models"].get(tag, {}), "eot_pgd20": round(acc_eot, 2)}})

    # ---- AutoAttack (rand 版, 带 EOT, 适合随机化防御) ----
    if not args.no_aa:
        print(f"\n========== 4) AutoAttack (version='rand', EOT={args.eot}) ==========")
        progress.set(phase="attacking", current_attack="AutoAttack(rand)")
        aa_src = os.path.join(ROOT, "auto-attack-master")
        if os.path.isdir(os.path.join(aa_src, "autoattack")):
            sys.path.insert(0, aa_src)
        from autoattack import AutoAttack
        model.eval()
        adv = AutoAttack(model, norm='Linf', eps=eps, version='rand',
                         device=device, verbose=False)
        if hasattr(adv.apgd, "eot_iter"):
            adv.apgd.eot_iter = args.eot
        if hasattr(adv.apgd_targeted, "eot_iter"):
            adv.apgd_targeted.eot_iter = args.eot
        t0 = time.time()
        x_adv = adv.run_standard_evaluation(x_aa, y_aa, bs=args.aa_batch)
        with torch.no_grad():
            rob = (model(x_adv).argmax(1) == y_aa).float().mean().item() * 100
        print(f"  [AA-rand] 鲁棒准确率 = {rob:.2f}%  ({time.time()-t0:.0f}s)")
        results["aa_rand"] = round(float(rob), 2)
        results["aa_rand_time_s"] = round(time.time() - t0, 1)
        progress.set(models={tag: {**progress.data["models"].get(tag, {}), "aa_rand": round(float(rob), 2)}})
        progress.log(f"AutoAttack(rand) 完成: 鲁棒率 {rob:.2f}%")

    report_path = os.path.join(REPORT_DIR, f"{tag}_experiment.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[report] 已保存 -> {report_path}")
    progress.set(status="done", phase="done", current_model="(完成)")
    progress.log("PixelDP 实验完成")


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
