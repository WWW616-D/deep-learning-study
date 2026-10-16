"""
随机平滑 (Randomized Smoothing) 认证实验
=======================================
关键设计: PixelDP 的训练就是"输入加噪 + CE", 与随机平滑的训练完全相同.
因此直接复用 PixelDP 训练出的权重, 在**同一批抽样**上同时计算两种证书:

  * PixelDP 证书: 平均概率向量 + Hoeffding 界 + DP 稳定性界   -> L_max
  * 随机平滑证书: 票率 + Clopper-Pearson 二项下界 + sigma*Phi^-1(p_A) -> R

唯一变量 = 认证数学 (e^{2eps} vs Phi^-1), 可以直接看出紧界能多换多少安全区.

用法:
  python rs_certify.py --sigma 0.2537            # 复用 pixeldp_L0.1 权重
  python rs_certify.py --sigma 0.0761            # 复用 pixeldp_L0.03 权重
  python rs_certify.py --sigma 0.2537 --quick
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
import torch.nn.functional as F
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
T = importlib.import_module("try")
import adv_train as A
import pixeldp_train as P

CKPT_DIR = T.CKPT_DIR
REPORT_DIR = T.REPORT_DIR


# ---------------- 随机平滑认证 ----------------
def cp_lower(k, n, alpha):
    """Clopper-Pearson 单侧下界 (置信度 1-alpha)."""
    if k <= 0:
        return 0.0
    if k >= n:
        return (1 - alpha) ** (1.0 / n)
    return float(stats.beta.ppf(alpha, k, n - k + 1))


def rs_radius(p_lb, sigma):
    """R = sigma * Phi^{-1}(p_lb)."""
    if p_lb <= 0.5:
        return 0.0
    return sigma * float(stats.norm.ppf(p_lb))


def pixeldp_radius(lb_top, ub_second, sigma, delta, mech="gaussian", L_hi=5.0, iters=60):
    """PixelDP Proposition 2: 解出满足 DP 稳定性界的最大 L."""
    c = math.sqrt(2 * math.log(1.25 / delta)) if mech == "gaussian" else math.sqrt(2)

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
def sample_both(f, x, y, sigma, n_draws, alpha, delta, chunk=25, group=250):
    """对每个样本抽样 n_draws 次, 同时累积【票数】和【概率和】, 一次抽样算两种证书."""
    N = x.size(0)
    K = 10
    hoeff = math.sqrt(math.log(2 * K / (1 - alpha)) / (2 * n_draws))
    out = {"rs_correct": [], "rs_radius": [], "rs_phat": [], "rs_plb": [],
           "pdp_radius": [], "rs_pred": [], "pdp_pred": []}
    for i in range(0, N, chunk):
        xb = x[i:i + chunk]
        b = xb.size(0)
        counts = torch.zeros(b, K)
        psum = torch.zeros(b, K)
        done = 0
        while done < n_draws:
            g = min(group, n_draws - done)
            z = torch.clamp(xb.repeat(g, 1, 1, 1)
                            + torch.randn(g * b, *xb.shape[1:]) * sigma, 0.0, 1.0)
            logits = f(z)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax(1).reshape(g, b)
            counts += F.one_hot(pred, K).float().sum(0)
            psum += probs.reshape(g, b, K).sum(0)
            done += g
        mean_p = psum / n_draws
        for j in range(b):
            # ---- 随机平滑证书 ----
            k = int(counts[j].max().item())
            top_rs = int(counts[j].argmax().item())
            p_hat = k / n_draws
            p_lb = cp_lower(k, n_draws, alpha)
            r_rs = rs_radius(p_lb, sigma)
            out["rs_phat"].append(p_hat)
            out["rs_plb"].append(p_lb)
            out["rs_radius"].append(r_rs)
            out["rs_pred"].append(top_rs)
            out["rs_correct"].append(int(top_rs == int(y[i + j].item())))
            # ---- PixelDP 证书 (同一批抽样) ----
            mp = mean_p[j]
            lb = torch.clamp(mp - hoeff, 0, 1)
            ub = torch.clamp(mp + hoeff, 0, 1)
            top = int(mp.argmax().item())
            second = torch.cat([ub[:top], ub[top + 1:]]).max().item()
            L = pixeldp_radius(float(lb[top].item()), second, sigma, delta)
            out["pdp_radius"].append(L)
            out["pdp_pred"].append(top)
    for kk in out:
        out[kk] = np.array(out[kk])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True, help="噪声标准差 (需与训练时一致)")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--ckpt", type=str, default=None, help="默认按 sigma 推断")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.samples, args.draws = 50, 500

    device = "cpu"
    sigma = args.sigma
    # 由 sigma 反推 L, 找到对应 checkpoint
    L_equiv = sigma * 1.0 / math.sqrt(2 * math.log(1.25 / args.delta))
    ckpt = args.ckpt or os.path.join(CKPT_DIR, f"pixeldp_L{round(L_equiv, 2)}_c1.pth")
    if not os.path.exists(ckpt):
        cands = [f for f in os.listdir(CKPT_DIR) if f.startswith("pixeldp_L")]
        print(f"[err] 找不到 {ckpt}; 现有: {cands}")
        return
    print(f"[env] sigma={sigma:.4f} (等价 L={L_equiv:.4f}) ckpt={os.path.basename(ckpt)}")
    print(f"[env] samples={args.samples} draws={args.draws} alpha={args.alpha}")

    model = P.PixelDPNet(T.NormalizeNet(A.C1Net()), sigma, "gaussian").to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    def f(z):                      # 确定性基础分类器: clamp 后送网络
        return model.base(z)

    _, _, testset = T.get_loaders(128)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(testset), size=args.samples, replace=False)
    x = torch.stack([testset[i][0] for i in idx]).to(device)
    y = torch.tensor([testset[i][1] for i in idx]).to(device)

    progress = T.Progress(os.path.join(REPORT_DIR, "progress.json"))
    progress.set(phase="attacking", current_model="(随机平滑认证)",
                 current_attack="RS 认证抽样")
    progress.log(f"开始 RS 认证 sigma={sigma} draws={args.draws} samples={args.samples}")

    t0 = time.time()
    res = sample_both(f, x, y, sigma, args.draws, args.alpha, args.delta)
    dt = time.time() - t0
    print(f"[cert] 抽样完成 {dt:.0f}s")

    thresholds = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    rs_acc, pdp_acc = {}, {}
    for Tt in thresholds:
        rs_acc[str(Tt)] = round(float(((res["rs_correct"] == 1) &
                                       (res["rs_radius"] >= Tt)).mean() * 100), 2)
        pdp_correct = (res["pdp_pred"] == y.numpy())
        pdp_acc[str(Tt)] = round(float((pdp_correct & (res["pdp_radius"] >= Tt)).mean() * 100), 2)

    print("\n===== 同一批权重 / 同一批抽样, 两种证书对比 =====")
    print(f"{'T':>6} | {'随机平滑 R=sigma*Phi^-1':>22} | {'PixelDP (DP 界)':>16}")
    print("-" * 54)
    for Tt in thresholds:
        print(f"{Tt:>6} | {rs_acc[str(Tt)]:>20.2f}% | {pdp_acc[str(Tt)]:>15.2f}%")

    rs_r = res["rs_radius"]
    pdp_r = res["pdp_radius"]
    print(f"\n平均半径:  随机平滑 = {rs_r.mean():.4f}   中位数 = {np.median(rs_r):.4f}"
          f"   正半径比例 = {(rs_r > 0).mean()*100:.1f}%")
    print(f"平均半径:  PixelDP  = {pdp_r.mean():.4f}   中位数 = {np.median(pdp_r):.4f}"
          f"   正半径比例 = {(pdp_r > 0).mean()*100:.1f}%")
    print(f"倍数(平均): {rs_r.mean()/max(pdp_r.mean(),1e-9):.1f}x")
    print(f"随机平滑票率: 平均 {res['rs_phat'].mean():.3f}, 最低 {res['rs_phat'].min():.3f}")
    print(f"干净准确率(投票预测): {(res['rs_correct'].mean())*100:.2f}%")

    results = {
        "sigma": sigma, "L_equiv": round(L_equiv, 4), "ckpt": os.path.basename(ckpt),
        "samples": args.samples, "draws": args.draws, "alpha": args.alpha,
        "delta": args.delta, "device": device,
        "clean_vote_acc": round(float(res["rs_correct"].mean() * 100), 2),
        "rs_mean_radius": round(float(rs_r.mean()), 4),
        "rs_median_radius": round(float(np.median(rs_r)), 4),
        "rs_positive_rate": round(float((rs_r > 0).mean() * 100), 2),
        "rs_mean_phat": round(float(res["rs_phat"].mean()), 4),
        "pdp_mean_radius": round(float(pdp_r.mean()), 4),
        "pdp_median_radius": round(float(np.median(pdp_r)), 4),
        "pdp_positive_rate": round(float((pdp_r > 0).mean() * 100), 2),
        "ratio_mean_radius": round(float(rs_r.mean() / max(pdp_r.mean(), 1e-9)), 2),
        "rs_certified_accuracy": rs_acc,
        "pdp_certified_accuracy": pdp_acc,
        "cert_time_s": round(dt, 1),
    }
    tag = f"rs_sigma{sigma:.4f}"
    rp = os.path.join(REPORT_DIR, f"{tag}_experiment.json")
    with open(rp, "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(f"\n[report] 已保存 -> {rp}")
    progress.set(status="done", phase="done", current_model="(完成)",
                 models={"rs": {"clean": results["clean_vote_acc"],
                                "mean_lmax": results["rs_mean_radius"],
                                "cert@0.1": rs_acc.get("0.1")}})
    progress.log(f"RS 认证完成: 平均半径 {results['rs_mean_radius']} (PixelDP {results['pdp_mean_radius']})")


if __name__ == "__main__":
    main()
