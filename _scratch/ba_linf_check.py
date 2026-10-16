"""补充检查: 边界攻击强化后的对抗样本, 是否仍满足原始威胁模型 (Linf <= 8/255)?
   同时报告 L2 / Linf 的变化, 避免"L2 变小但 Linf 越界"的误导性结论."""
import os, sys, importlib, time, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\py\code")
T = importlib.import_module("try")
import adv_train as A
import pixeldp_train as P
import boundary_experiment as B
import numpy as np
import torch

device = "cpu"
eps = 8/255.0
N, EOT, STEPS, MAXR = 200, 10, 1500, 50

specs = [
    ("对抗训练 C1", "adv", os.path.join(T.CKPT_DIR, "adv_train_c1.pth"), 0.0, True),
    ("PixelDP σ=0.25", "pixeldp", os.path.join(T.CKPT_DIR, "pixeldp_L0.1_c1.pth"), 0.2537, False),
    ("噪声训练 σ=0.51 (RS)", "rs", os.path.join(T.CKPT_DIR, "pixeldp_L0.2_c1.pth"), 0.5075, False),
]
targets = []
for name, key, ckpt, sigma, det in specs:
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    if det:
        m = T.NormalizeNet(A.C1Net()).to(device); m.load_state_dict(sd)
    else:
        tmp = P.PixelDPNet(T.NormalizeNet(A.C1Net()), sigma, "gaussian").to(device)
        tmp.load_state_dict(sd); m = tmp.base
    m.eval()
    targets.append(B.Target(name, key, m, sigma=sigma, k_decision=EOT, deterministic=det))

_, _, testset = T.get_loaders(128)
rng = np.random.RandomState(0)
idx = rng.choice(len(testset), size=N, replace=False)
x = torch.stack([testset[i][0] for i in idx]).to(device)
y = torch.tensor([testset[i][1] for i in idx]).to(device)

sys.path.insert(0, os.path.join(T.ROOT, "auto-attack-master"))
from autoattack import AutoAttack

out = {}
for tg in targets:
    print(f"\n===== {tg.name} =====")
    version = "standard" if tg.deterministic else "rand"
    aa = AutoAttack(tg.model, norm='Linf', eps=eps, version=version, device=device, verbose=False)
    if version == "rand" and hasattr(aa.apgd, "eot_iter"):
        aa.apgd.eot_iter = EOT
    if hasattr(aa.square, "n_queries"):
        aa.square.n_queries = 1500
    x_adv = aa.run_standard_evaluation(x, y, bs=100)
    pred = tg.decide_k(x_adv, 50); clean = tg.decide_k(x, 50)
    ok = (pred != y) & (clean == y)
    print(f"  500抽样复核后 AA 有效对抗样本: {int(ok.sum())}/{N}")
    if ok.sum() == 0:
        continue
    sel = torch.nonzero(ok).squeeze(1)[:MAXR]
    xb, yb, ab = x[sel], y[sel], x_adv[sel]
    l2_before = B.l2(ab - xb); linf_before = B.linf(ab - xb)
    print(f"  AA 扰动: L2中位 {float(l2_before.median()):.4f}, Linf中位 {float(linf_before.median()):.4f} "
          f"({float(linf_before.median())*255:.1f}/255)")
    t0 = time.time()
    adv_m, d_m = B.boundary_attack(tg, xb, yb, ab, STEPS)
    dt = time.time() - t0
    ver = B.verify(tg, adv_m, yb, 50)
    l2_after = B.l2(adv_m - xb); linf_after = B.linf(adv_m - xb)
    v = ver
    print(f"  BA 强化后 ({int(v.sum())} 个复核通过, {dt:.0f}s):")
    if v.any():
        print(f"    L2  中位 {float(l2_before[v].median()):.4f} -> {float(l2_after[v].median()):.4f} "
              f"(比 {float((l2_after[v]/l2_before[v]).median()):.3f})")
        print(f"    Linf中位 {float(linf_before[v].median())*255:.1f}/255 -> "
              f"{float(linf_after[v].median())*255:.1f}/255")
        within = (linf_after[v] <= eps + 1e-6)
        print(f"    仍满足 Linf<=8/255 的比例: {int(within.sum())}/{int(v.sum())} "
              f"({float(within.float().mean())*100:.1f}%)")
    out[tg.key] = {
        "name": tg.name, "n_aa": int(ok.sum()), "n_verified": int(v.sum()),
        "l2_before_median": round(float(l2_before.median()), 4),
        "linf_before_median_255": round(float(linf_before.median())*255, 2),
        "l2_after_median": round(float(l2_after[v].median()), 4) if v.any() else None,
        "linf_after_median_255": round(float(linf_after[v].median())*255, 2) if v.any() else None,
        "ratio_l2": round(float((l2_after[v]/l2_before[v]).median()), 3) if v.any() else None,
        "frac_within_linf_budget": round(float((linf_after[v] <= eps+1e-6).float().mean()*100), 1) if v.any() else None,
        "steps": STEPS, "time_s": round(dt, 1),
    }

rp = os.path.join(T.REPORT_DIR, "ba_linf_check.json")
with open(rp, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n[report] -> {rp}")
