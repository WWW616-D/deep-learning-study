"""
决策式对抗攻击 (Boundary Attack) 实验
====================================
论文: Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box
      Machine Learning Models (Brendel, Rauber, Bethge; ICLR 2018)

算法 (论文 Algorithm 1):
  - 从"已是对抗样本"的点出发, 每步: (1) 沿边界做正交随机扰动 (尺度 delta*d),
    (2) 朝原图走一小步 (尺度 eps*d), 只要仍是对抗样本就接受 -> 逐步最小化 L2 距离
  - 动态调 delta/eps: 正交步的成功率应约 50%, 朝原图步的成功率应高于阈值
  - 只用模型最终标签 (决策式/黑盒), 不需要梯度或置信度

本轮实验流程:
  Round 1: AutoAttack (Linf eps=8/255) 攻破三个防御模型 -> 鲁棒率 R1 + 对抗样本
           (确定性模型用 standard; 随机化模型用 rand+EOT, 各自最严格协议)
  Round 2: 边界攻击 (决策式黑盒)
     (a) 强化: 对 AA 已攻破的样本, 从 AA 对抗样本出发做扰动最小化 -> 最小 L2
     (b) 补刀: 对 AA 的幸存者(仍正确), 从"大扰动误分类点"出发尝试攻破 -> R2
  对比 R1 vs R2

用法:
  python boundary_experiment.py            # 完整
  python boundary_experiment.py --quick    # 冒烟
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import importlib
T = importlib.import_module("try")
import adv_train as A
import pixeldp_train as P

CKPT_DIR = T.CKPT_DIR
REPORT_DIR = T.REPORT_DIR
ROOT = T.ROOT
PROG = os.path.join(REPORT_DIR, "ba_progress.json")


# ---------------- 进度上报 ----------------
class Prog:
    def __init__(self, path):
        self.path = path
        self.d = {"status": "running", "phase": "初始化", "started_at": time.strftime("%H:%M:%S"),
                  "updated_at": "", "step": 0, "total_steps": 0, "models": [],
                  "logs": [], "current": None, "query_count": 0}
        self.save()

    def save(self):
        self.d["updated_at"] = time.strftime("%H:%M:%S")
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.d, f, ensure_ascii=False)
        os.replace(tmp, self.path)

    def set(self, **kw):
        self.d.update(kw)
        self.save()

    def log(self, msg):
        self.d["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self.d["logs"] = self.d["logs"][-80:]
        self.save()

    def model(self, key, **kw):
        for m in self.d["models"]:
            if m["key"] == key:
                m.update(kw)
                self.save()
                return
        self.d["models"].append({"key": key, **kw})
        self.save()


# ---------------- 攻击目标封装 ----------------
class Target:
    """把模型包成决策函数: 随机化模型用 K 次抽样多数投票."""
    def __init__(self, name, key, model, sigma=0.0, k_decision=10, deterministic=False):
        self.name, self.key = name, key
        self.model, self.sigma = model, sigma
        self.k = 1 if deterministic else k_decision
        self.deterministic = deterministic
        self.queries = 0

    @torch.no_grad()
    def decide_k(self, x, k):
        """用指定的抽样次数 k 做多数投票 (k=1 时为单次前向)."""
        self.queries += x.size(0) * k
        votes = None
        for _ in range(k):
            z = x
            if not self.deterministic:
                z = torch.clamp(x + torch.randn_like(x) * self.sigma, 0, 1)
            lab = self.model(z).argmax(1)
            oh = F.one_hot(lab, 10).float()
            votes = oh if votes is None else votes + oh
        return votes.argmax(1)

    @torch.no_grad()
    def decide(self, x):
        """返回标签 [B]. 随机化模型: K 次抽样多数投票 (黑盒攻击者只能看标签)."""
        if self.deterministic:
            self.queries += x.size(0)
            return self.model(x).argmax(1)
        return self.decide_k(x, self.k)


@torch.no_grad()
def verify(target, x, y, k_verify=50):
    """用较高抽样次数复核对抗样本是否真的成立 (防止随机化模型上的投票噪声假成功)."""
    return (target.decide_k(x, k_verify) != y)


# ---------------- 边界攻击 (批量化) ----------------
@torch.no_grad()
def boundary_attack(target, x_orig, y_orig, init_adv, steps, delta=0.01, eps_step=0.01,
                    step_adapt=0.9, min_delta=1e-4, max_delta=0.05, min_eps=1e-4, max_eps=1.0,
                    on_step=None):
    """批量边界攻击: 最小化 L2 距离, 保持样本仍是对抗样本.
    x_orig: [B,3,32,32] 原图; init_adv: [B,...] 初始对抗样本; 返回 (最终对抗样本, 距离)."""
    B = x_orig.size(0)
    adv = init_adv.clone()
    prev = adv.clone()
    dv = torch.full((B,), float(delta))
    ev = torch.full((B,), float(eps_step))
    for it in range(steps):
        direction = (x_orig - adv).flatten(1)
        dn = direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
        u = direction / dn
        eta = torch.randn_like(adv).flatten(1)
        eta = eta - (eta * u).sum(1, keepdim=True) * u          # 正交于"朝原图"方向
        v = (adv - prev).flatten(1)
        vn = v.norm(dim=1, keepdim=True)
        act = (vn.squeeze(1) > 1e-12)
        if act.any():
            vu = v / vn.clamp_min(1e-12)
            eta[act] = eta[act] - (eta[act] * vu[act]).sum(1, keepdim=True) * vu[act]
        en = eta.norm(dim=1, keepdim=True).clamp_min(1e-12)
        eta = (eta / en * (dv.view(-1, 1) * dn)).reshape_as(adv)
        cand1 = torch.clamp(adv + eta, 0, 1)
        cand2 = torch.clamp(cand1 + ev.view(-1, 1, 1, 1) * (x_orig - cand1), 0, 1)
        labs = target.decide(torch.cat([cand1, cand2], 0))
        ok1 = (labs[:B] != y_orig)
        ok2 = (labs[B:] != y_orig)
        # 接受正交步, 再尝试朝原图步
        adv = torch.where(ok1.view(-1, 1, 1, 1), cand1, adv)
        good2 = ok1 & ok2
        adv = torch.where(good2.view(-1, 1, 1, 1), cand2, adv)
        prev = torch.where(ok1.view(-1, 1, 1, 1), prev, adv)
        # 动态调步长 (逐样本)
        dv = torch.where(ok1, torch.clamp(dv / step_adapt, max=max_delta),
                         torch.clamp(dv * step_adapt, min=min_delta))
        ev = torch.where(good2, torch.clamp(ev / step_adapt, max=max_eps),
                         torch.clamp(ev * step_adapt, min=min_eps))
        if on_step is not None and (it % 50 == 0 or it == steps - 1):
            on_step(it, adv, x_orig)
    dist = (adv - x_orig).flatten(1).norm(dim=1)
    return adv, dist


def l2(t):
    return t.flatten(1).norm(dim=1)


def linf(t):
    return t.flatten(1).abs().max(dim=1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--aa-batch", type=int, default=100)
    ap.add_argument("--eps", type=float, default=8.0)
    ap.add_argument("--eot", type=int, default=10)
    ap.add_argument("--ba-steps-refine", type=int, default=1500, help="(a) 扰动最小化步数")
    ap.add_argument("--ba-steps-survivor", type=int, default=4000, help="(b) 攻击幸存者步数")
    ap.add_argument("--max-refine", type=int, default=100)
    ap.add_argument("--max-survivor", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.samples = 50
        args.ba_steps_refine = 100
        args.ba_steps_survivor = 200
        args.max_refine = 20
        args.max_survivor = 20
        args.eot = 3

    device = "cpu"
    eps = args.eps / 255.0
    prog = Prog(PROG)
    prog.set(phase="加载模型", total_steps=3)
    print(f"[env] samples={args.samples} eps={args.eps}/255 eot={args.eot} "
          f"ba_refine={args.ba_steps_refine} ba_survivor={args.ba_steps_survivor}")

    # ---- 三个防御模型 ----
    specs = [
        ("对抗训练 C1", "adv", os.path.join(CKPT_DIR, "adv_train_c1.pth"), 0.0, True),
        ("PixelDP σ=0.25", "pixeldp", os.path.join(CKPT_DIR, "pixeldp_L0.1_c1.pth"), 0.2537, False),
        ("噪声训练 σ=0.51 (RS)", "rs", os.path.join(CKPT_DIR, "pixeldp_L0.2_c1.pth"), 0.5075, False),
    ]
    targets = []
    for name, key, ckpt, sigma, det in specs:
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        if det:
            # adv_train_c1.pth 存的是 NormalizeNet(C1Net()): keys = net.*/mean/std
            m = T.NormalizeNet(A.C1Net()).to(device)
            m.load_state_dict(sd)
        else:
            # pixeldp_*.pth 存的是 PixelDPNet: keys = base.* -> 取出 base 作为确定性分类器
            tmp = P.PixelDPNet(T.NormalizeNet(A.C1Net()), sigma, "gaussian").to(device)
            tmp.load_state_dict(sd)
            m = tmp.base
        m.eval()
        targets.append(Target(name, key, m, sigma=sigma, k_decision=args.eot, deterministic=det))
        prog.model(key, name=name, status="loaded", aa_robust=None, survivors=None)
        print(f"[load] {name} <- {os.path.basename(ckpt)}")

    _, _, testset = T.get_loaders(128)
    rng = np.random.RandomState(args.seed)
    idx = rng.choice(len(testset), size=args.samples, replace=False)
    x = torch.stack([testset[i][0] for i in idx]).to(device)
    y = torch.tensor([testset[i][1] for i in idx]).to(device)

    sys.path.insert(0, os.path.join(ROOT, "auto-attack-master"))
    from autoattack import AutoAttack

    results = {"samples": args.samples, "eps": args.eps, "eot": args.eot,
               "ba_steps_refine": args.ba_steps_refine,
               "ba_steps_survivor": args.ba_steps_survivor, "models": {}}

    for ti, tg in enumerate(targets):
        print(f"\n{'='*70}\n[{tg.name}]  ({ti+1}/{len(targets)})\n{'='*70}")
        mres = {"name": tg.name}
        prog.set(phase=f"Round 1: AutoAttack · {tg.name}", current=tg.name, step=ti)
        prog.log(f"开始 {tg.name}: AutoAttack (eps={args.eps}/255, EOT={args.eot})")

        # ---------- Round 1: AutoAttack ----------
        version = "standard" if tg.deterministic else "rand"
        adv_aa = AutoAttack(tg.model, norm='Linf', eps=eps, version=version,
                            device=device, verbose=False)
        if version == "rand" and hasattr(adv_aa.apgd, "eot_iter"):
            adv_aa.apgd.eot_iter = args.eot
        if hasattr(adv_aa.square, "n_queries"):
            adv_aa.square.n_queries = 1500
        t0 = time.time()
        x_adv = adv_aa.run_standard_evaluation(x, y, bs=args.aa_batch)
        aa_time = time.time() - t0
        with torch.no_grad():
            pred_aa = tg.decide_k(x_adv, 50)      # 高抽样复核 AA 结果
            clean_pred = tg.decide_k(x, 50)       # 干净图预测 (高抽样)
        clean_ok = (clean_pred == y)              # 干净图就分错的样本必须排除
        broken = (pred_aa != y)
        r1 = float((~broken).float().mean() * 100)
        print(f"  [AA-{version}] 鲁棒率 R1 = {r1:.2f}%  ({aa_time:.0f}s, 攻破 {int(broken.sum())}/{args.samples})")
        print(f"  [AA] 干净图正确 {int(clean_ok.sum())}/{args.samples}; "
              f"其中被 AA 攻破 {int((broken & clean_ok).sum())}")
        l2_aa = l2(x_adv - x)[broken & clean_ok]
        med_aa_l2 = float(l2_aa.median()) if l2_aa.numel() > 0 else float('nan')
        print(f"  [AA] 成功扰动 L2: 中位数 {med_aa_l2:.4f}, 均值 {float(l2_aa.mean()):.4f}")
        prog.model(tg.key, aa_robust=round(r1, 2), aa_time=round(aa_time, 1),
                   aa_broken=int(broken.sum()), survivors=int((~broken).sum()),
                   clean_correct=int(clean_ok.sum()),
                   aa_l2_median=round(med_aa_l2, 4), aa_version=version, status="AA 完成")
        prog.log(f"{tg.name}: AA 鲁棒率 {r1:.2f}%, 中位扰动 L2 {med_aa_l2:.4f}, "
                 f"干净正确 {int(clean_ok.sum())}")

        # ---------- Round 2(a): 强化 = 扰动最小化 (只用"干净图正确 且 被AA攻破"的样本) ----------
        prog.set(phase=f"Round 2a: 边界攻击·扰动最小化 · {tg.name}")
        idx_broken = torch.nonzero(broken & clean_ok).squeeze(1)
        if idx_broken.numel() > 0:
            sel = idx_broken[:args.max_refine]
            xb, yb, ab = x[sel], y[sel], x_adv[sel]
            lab0 = tg.decide(ab)
            keep = (lab0 != yb)
            xb, yb, ab = xb[keep], yb[keep], ab[keep]
            if xb.size(0) > 0:
                prog.log(f"{tg.name}: 扰动最小化 {xb.size(0)} 个样本, {args.ba_steps_refine} 步")
                t0 = time.time()
                adv_min, d_min = boundary_attack(tg, xb, yb, ab, args.ba_steps_refine)
                dt = time.time() - t0
                # 高抽样复核: 只统计复核后仍成立的样本
                ver = verify(tg, adv_min, yb, 50)
                d_before = l2(ab - xb)
                nv = int(ver.sum())
                if nv > 0:
                    d_ok = d_min[ver]
                    ratio = float((d_ok / d_before[ver].clamp_min(1e-12)).median())
                    ref_before = float(d_before[ver].median())
                    ref_after = float(d_ok.median())
                else:
                    ratio, ref_before, ref_after = float('nan'), float('nan'), float('nan')
                print(f"  [BA-refine] {xb.size(0)} 样本 (复核通过 {nv}): "
                      f"L2 {float(d_before.median()):.4f} -> {ref_after:.4f} "
                      f"(中位比 {ratio:.3f}, {dt:.0f}s)")
                prog.model(tg.key, refine_n=int(xb.size(0)), refine_verified=nv,
                           refine_before=round(ref_before, 4) if nv else None,
                           refine_after=round(ref_after, 4) if nv else None,
                           refine_ratio=round(ratio, 3) if nv else None,
                           refine_time=round(dt, 1))
                prog.log(f"{tg.name}: 扰动最小化 {float(d_before.median()):.4f} -> {ref_after:.4f} (复核 {nv})")
                mres["refine"] = {"n": int(xb.size(0)), "verified": nv,
                                  "l2_before_median": round(ref_before, 4) if nv else None,
                                  "l2_after_median": round(ref_after, 4) if nv else None,
                                  "ratio_median": round(ratio, 3) if nv else None,
                                  "time_s": round(dt, 1)}
            else:
                print("  [BA-refine] AA 样本二次判定后无对抗样本, 跳过")
        else:
            print("  [BA-refine] 无 AA 成功样本, 跳过")

        # ---------- Round 2(b): 攻击 AA 幸存者 (只用干净图正确的样本) ----------
        prog.set(phase=f"Round 2b: 边界攻击·攻击幸存者 · {tg.name}")
        idx_surv = torch.nonzero(~broken & clean_ok).squeeze(1)
        surv_broken = 0
        d_surv_median = float('nan')
        if idx_surv.numel() > 0:
            sel = idx_surv[:args.max_survivor]
            xs, ys = x[sel], y[sel]
            # 初始化: 找"最小的噪声尺度"使其误分类 (起点越近, 边界攻击越有优势)
            init = torch.zeros_like(xs)
            got = torch.zeros(xs.size(0), dtype=torch.bool)
            for s in (0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.5, 0.7, 0.9):
                need = ~got
                if not need.any():
                    break
                cand = torch.clamp(xs + torch.randn_like(xs) * s, 0, 1)
                lab = tg.decide_k(cand, 50)     # 起始点也用高抽样复核
                good = (lab != ys) & need
                init[good] = cand[good]
                got = got | good
            xs, ys, init = xs[got], ys[got], init[got]
            if xs.size(0) > 0:
                prog.log(f"{tg.name}: 攻击 {xs.size(0)} 个幸存者, {args.ba_steps_survivor} 步")
                t0 = time.time()
                adv_s, d_s = boundary_attack(tg, xs, ys, init, args.ba_steps_survivor)
                dt = time.time() - t0
                # 高抽样复核
                ver = verify(tg, adv_s, ys, 50)
                d_surv_median = float(d_s[ver].median()) if ver.any() else float('nan')
                # 预算: 1.7442 = Linf eps=8/255 球的外接 L2 半径 (最宽松读法)
                surv_broken = int((ver & (d_s <= 1.7442)).sum())
                n_05 = int((ver & (d_s <= 0.5)).sum())
                n_003 = int((ver & (d_s <= 0.0314)).sum())
                print(f"  [BA-survivor] {xs.size(0)} 幸存者 (复核成立 {int(ver.sum())}): "
                      f"最小 L2 中位数 {d_surv_median:.4f} | ≤1.7442 的 {surv_broken} 个, "
                      f"≤0.5 的 {n_05} 个, ≤0.0314(严格) 的 {n_003} 个 ({dt:.0f}s)")
                prog.model(tg.key, surv_n=int(xs.size(0)), surv_verified=int(ver.sum()),
                           surv_min_l2_median=round(d_surv_median, 4) if ver.any() else None,
                           surv_in_budget=surv_broken, surv_time=round(dt, 1))
                mres["survivor"] = {
                    "n": int(xs.size(0)), "verified": int(ver.sum()),
                    "min_l2_median": round(d_surv_median, 4) if ver.any() else None,
                    "n_within_1.744": surv_broken, "n_within_0.5": n_05,
                    "n_within_0.0314": n_003, "time_s": round(dt, 1)}
            else:
                print("  [BA-survivor] 无法构造初始对抗点, 跳过")

        n_surv_total = int((~broken & clean_ok).sum())
        r2 = float((int((~broken).sum()) - surv_broken) / args.samples * 100)
        print(f"  ==> {tg.name}: R1(AA) = {r1:.2f}%  ->  R2(AA+BA) = {r2:.2f}%")
        prog.model(tg.key, r2=round(r2, 2), status="完成", ba_queries=int(tg.queries))
        prog.log(f"{tg.name} 完成: R1={r1:.2f}% -> R2={r2:.2f}%")
        mres.update({
            "aa_version": version, "aa_robust": round(r1, 2),
            "aa_broken": int(broken.sum()), "survivors": n_surv_total,
            "aa_l2_median": round(med_aa_l2, 4), "ba_robust": round(r2, 2),
            "survivors_broken_by_ba": surv_broken, "ba_queries": int(tg.queries),
        })
        results["models"][tg.key] = mres

    rp = os.path.join(REPORT_DIR, "boundary_experiment.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[report] 已保存 -> {rp}")
    prog.set(status="done", phase="全部完成", step=3)
    prog.log("实验完成")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            with open(PROG, encoding="utf-8") as f:
                d = json.load(f)
            d["status"] = "error"
            d["logs"] = d.get("logs", []) + [traceback.format_exc()[-800:]]
            with open(PROG, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False)
        except Exception:
            pass
        raise
