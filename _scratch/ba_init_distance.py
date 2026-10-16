"""补测 Round 2b 的初始距离: 幸存者初始化时能从多远起步 (报告里漏记的关键数字)."""
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
N, EOT = 500, 10
LADDER = (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9)

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
    broken = (tg.decide_k(x_adv, 50) != y)
    clean_ok = (tg.decide_k(x, 50) == y)
    surv = (~broken & clean_ok)
    print(f"  AA 鲁棒率 {float((~broken).float().mean()*100):.2f}% | "
          f"干净正确 {int(clean_ok.sum())} | 幸存者(干净正确且AA未破) {int(surv.sum())}")
    xs, ys = x[surv], y[surv]
    # ---- 复现 Round 2b 的初始化阶梯 ----
    init = torch.zeros_like(xs)
    got = torch.zeros(xs.size(0), dtype=torch.bool)
    used_s = torch.zeros(xs.size(0))
    for s in LADDER:
        need = ~got
        if not need.any():
            break
        cand = torch.clamp(xs + torch.randn_like(xs) * s, 0, 1)
        lab = tg.decide_k(cand, 50)
        good = (lab != ys) & need
        init[good] = cand[good]
        used_s[good] = s
        got = got | good
    if got.sum() == 0:
        print("  无样本能初始化"); continue
    xi, yi, ii, si = xs[got], ys[got], init[got], used_s[got]
    d_init = B.l2(ii - xi)
    print(f"  可初始化 {xi.size(0)}/{xs.size(0)} 个")
    print(f"  初始化用的噪声尺度: 最小 {float(si.min()):.2f}, 中位 {float(si.median()):.2f}, 最大 {float(si.max()):.2f}")
    print(f"  ==> 初始 L2 距离: 中位 {float(d_init.median()):.2f}, 最小 {float(d_init.min()):.2f}, 最大 {float(d_init.max()):.2f}")
    print(f"      (要达到 Linf 8/255 预算 L2=1.7442, 还差 {float(d_init.median()/1.7442):.1f} 倍)")
    out[tg.key] = {
        "name": tg.name, "aa_robust": round(float((~broken).float().mean()*100), 2),
        "survivors": int(surv.sum()), "init_ok": int(got.sum()),
        "noise_scale_median": round(float(si.median()), 3),
        "init_l2_median": round(float(d_init.median()), 2),
        "init_l2_min": round(float(d_init.min()), 2),
        "init_l2_max": round(float(d_init.max()), 2),
        "gap_to_budget": round(float(d_init.median()/1.7442), 1),
    }

rp = os.path.join(T.REPORT_DIR, "ba_init_distance.json")
with open(rp, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n[report] -> {rp}")
