"""
CIFAR-10 对抗防御 vs AutoAttack 实验
====================================
用本地 CIFAR-10 数据集训练 LeNet 模型, 施加两类防御, 再用 AutoAttack 攻击:

  模型 0 baseline   : 普通交叉熵训练 (无防御)
  模型 1 distill    : 防御蒸馏 (教师 T=1 正常训练 + 学生 T=20 软标签,
                        lr 补偿 + 无 wd, 参考 Papernot 2016 / Hinton 2015 / ART)
  模型 2 mask_sat   : 梯度遮蔽-饱和输出网络 (log_sigmoid 输出, 梯度消失型)
  模型 3 mask_prep  : 梯度遮蔽-位深缩减预处理 (5bit 量化, 梯度粉碎型)

攻击:
  * AutoAttack standard (Linf, eps=8/255): APGD-CE / APGD-DLR / FAB / Square
  * PGD-20 白盒 (对照)
  * BPDA-PGD-20 (针对两类梯度遮蔽的自适应攻击, 用可微代理模型算梯度)
  * 迁移攻击: 在 baseline 上生成的 PGD 对抗样本迁移到各防御模型

用法:
  python try.py            # 完整实验 (40 epochs, AA 1000 样本)
  python try.py --quick    # 冒烟测试 (2 epochs, AA 50 样本)
  python try.py --reuse    # 复用已保存的 checkpoint, 跳过训练
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 规避 OMP 库冲突

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# ---------------- 路径 ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_ROOT = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(HERE, "checkpoints")
REPORT_DIR = os.path.join(ROOT, "reports")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# ---------------- 实时进度上报 (供进度网页轮询) ----------------
class Progress:
    """把实验状态写到一个 JSON, 由 reports/progress.html 轮询展示。"""
    def __init__(self, path):
        self.path = path
        self.data = {
            "status": "running", "phase": "init", "started_at": _now(),
            "updated_at": _now(), "current_model": "", "current_epoch": 0,
            "total_epochs": 0, "current_loss": None, "current_acc": None,
            "eta_s": None, "models": {}, "attacks": [], "logs": [],
            "error": None,
        }
        self._epoch_times = []
        self._t0_phase = time.time()

    def set(self, **kw):
        self.data.update(kw)
        self.data["updated_at"] = _now()
        self.save()

    def log(self, msg):
        self.data["logs"].append(f"[{_now()}] {msg}")
        self.data["logs"] = self.data["logs"][-60:]
        self.set()

    def note_epoch_done(self, dt):
        self._epoch_times.append(dt)
        self._epoch_times = self._epoch_times[-20:]

    def estimate_eta(self):
        if not self._epoch_times:
            return None
        avg = sum(self._epoch_times) / len(self._epoch_times)
        remaining = self.data["total_epochs"] - self.data["current_epoch"]
        return avg * max(remaining, 0)

    def save(self):
        try:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False)
            os.replace(tmp, self.path)
        except Exception:
            pass


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------- 模型定义 ----------------
class LeNet(nn.Module):
    """与 data/cifar_net.pth 同架构的 LeNet, CPU 上训练快"""
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
        return self.fc3(x)


class NormalizeNet(nn.Module):
    """把归一化包进模型, 使攻击作用于完整 pipeline (输入范围 [0,1])"""
    def __init__(self, net, mean=0.5, std=0.5):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.full((1, 3, 1, 1), mean))
        self.register_buffer("std", torch.full((1, 3, 1, 1), std))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


def bit_depth_reduce(x, bits=5):
    """把 [0,1] 输入量化到 bits 位 (保留高 bits 位)。round 梯度几乎处处为 0。"""
    step = 2 ** (8 - bits)
    return torch.round(torch.round(x * 255) / step) * step / 255.0


class PreprocessNet(NormalizeNet):
    """梯度遮蔽变体 B: 训练/推理前先做位深缩减 (梯度粉碎)"""
    def __init__(self, net, bits=5):
        super().__init__(net)
        self.bits = bits

    def forward(self, x):
        return self.net(bit_depth_reduce(x, self.bits))


class SaturatedNet(NormalizeNet):
    """梯度遮蔽变体 A: 饱和输出网络, logits 经 log_sigmoid 饱和 (梯度消失)"""
    def __init__(self, net, gain=1.0):
        super().__init__(net)
        self.gain = gain

    def forward(self, x):
        return F.logsigmoid(self.net(x) * self.gain)


def make_model(kind, seed=0):
    torch.manual_seed(seed)
    if kind == "baseline":
        return NormalizeNet(LeNet())
    if kind == "distill":
        return NormalizeNet(LeNet())
    if kind == "mask_sat":
        return SaturatedNet(LeNet(), gain=1.0)
    if kind == "mask_prep":
        return PreprocessNet(LeNet(), bits=5)
    raise ValueError(kind)


# ---------------- 数据 ----------------
def get_loaders(batch_size):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])
    test_tf = T.Compose([T.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True, download=False, transform=train_tf)
    testset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, download=False, transform=test_tf)
    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=batch_size,
                             shuffle=False, num_workers=0)
    return train_loader, test_loader, testset


# ---------------- 训练与评估 ----------------
def evaluate(model, loader, device, desc=""):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    print(f"    [clean] {desc or model.__class__.__name__}: {acc:.2f}% ({correct}/{total})")
    return acc


def train_model(model, train_loader, eval_loader, device, epochs, lr,
                T=1.0, teacher=None, tag="", wd=5e-4, progress=None):
    """teacher=None, T=1 -> 普通 CE; T>1 -> 高温 CE; teacher 非空 -> 软标签蒸馏。
    调研结论 (Papernot 2016 / Hinton 2015 / ART): 高温使梯度缩小 T 倍,
    需补偿学习率并去掉 weight decay, 否则权重被压向 0 学不动。"""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = n = 0
        t_ep = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            z = model(x)
            if teacher is None:
                loss = F.cross_entropy(z / T, y) if T != 1.0 else F.cross_entropy(z, y)
            else:
                with torch.no_grad():
                    p = F.softmax(teacher(x) / T, dim=1)   # 教师软标签
                loss = -(p * F.log_softmax(z / T, dim=1)).sum(1).mean()
            loss.backward()
            opt.step()
            run_loss += loss.item() * x.size(0)
            n += x.size(0)
        sched.step()
        acc = evaluate(model, eval_loader, device, desc=f"{tag} ep{ep}")
        if progress is not None:
            progress.set(current_model=tag, current_epoch=ep,
                         total_epochs=epochs, current_loss=run_loss / n,
                         current_acc=acc)
            progress.note_epoch_done(time.time() - t_ep)
            progress.set(eta_s=progress.estimate_eta())
    return model


# ---------------- 攻击 ----------------
def pgd_attack(model, x, y, eps, alpha, iters, surrogate=None,
               rand_start=True, device="cpu"):
    """Linf PGD。surrogate=None 为白盒; 否则 BPDA: 真实前向 + 代理梯度。"""
    model.eval()
    x0 = x.clone().detach()
    delta = torch.zeros_like(x0)
    if rand_start:
        delta.uniform_(-eps, eps)
        delta = torch.clamp(delta, -x0, 1 - x0)
    for _ in range(iters):
        delta = delta.detach().requires_grad_(True)
        x_adv = torch.clamp(x0 + delta, 0, 1)
        if surrogate is None:
            loss = F.cross_entropy(model(x_adv), y)
            grad = torch.autograd.grad(loss, delta)[0]
        else:
            with torch.no_grad():
                model(x_adv)                       # 真实前向 (只用于保证路径一致)
            x_s = x_adv.detach().requires_grad_(True)
            loss_s = F.cross_entropy(surrogate(x_s), y)
            grad = torch.autograd.grad(loss_s, x_s)[0]   # 代理梯度
        delta = delta.detach() + alpha * grad.sign()
        delta = torch.clamp(delta, -eps, eps)
        delta = torch.clamp(delta, -x0, 1 - x0)
    return torch.clamp(x0 + delta, 0, 1)


def attack_eval(model, x, y, eps, alpha, iters, surrogate=None, device="cpu"):
    x_adv = pgd_attack(model, x, y, eps, alpha, iters, surrogate, device=device)
    with torch.no_grad():
        acc = (model(x_adv).argmax(1) == y).float().mean().item() * 100
    return acc


def run_autoattack(model, x, y, eps, aa_batch, square_queries, aa_iters, device):
    # autoattack 库: 源码放在项目根目录 auto-attack-master/ (网络受限无法 pip 安装)
    aa_src = os.path.join(ROOT, "auto-attack-master")
    if os.path.isdir(os.path.join(aa_src, "autoattack")):
        sys.path.insert(0, aa_src)
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm='Linf', eps=eps, version='standard',
                           device=device, verbose=False)
    # CPU 上收敛预算: 只压缩最贵的 Square 查询数, 其余保持 standard 预设
    sq = getattr(adversary, "square", None)
    if sq is not None and hasattr(sq, "n_queries"):
        sq.n_queries = square_queries
    model.eval()
    t0 = time.time()
    x_adv = adversary.run_standard_evaluation(x, y, bs=aa_batch)
    with torch.no_grad():
        robust_acc = (model(x_adv).argmax(1) == y).float().mean().item() * 100
    return robust_acc, time.time() - t0


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--student-lr", type=float, default=4.0,
                    help="蒸馏学生学习率 (补偿 T 导致的梯度缩小, wd=0)")
    ap.add_argument("--distill-t", type=float, default=20.0)
    ap.add_argument("--aa-samples", type=int, default=1000)
    ap.add_argument("--aa-batch", type=int, default=100)
    ap.add_argument("--eps", type=float, default=8.0, help="Linf epsilon (单位 1/255)")
    ap.add_argument("--aa-iters", type=int, default=100, help="APGD/FAB 迭代数")
    ap.add_argument("--aa-square-queries", type=int, default=1500, help="Square 查询数")
    ap.add_argument("--pgd-iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="冒烟测试")
    ap.add_argument("--reuse", action="store_true", help="复用已有 checkpoint")
    ap.add_argument("--no-aa", action="store_true", help="跳过 AutoAttack")
    args = ap.parse_args()

    if args.quick:
        args.epochs, args.aa_samples = 2, 50
        args.aa_square_queries = 200
        args.pgd_iters = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    eps = args.eps / 255.0
    alpha = 2.0 / 255.0
    print(f"[env] device={device} epochs={args.epochs} eps={args.eps}/255 "
          f"aa_samples={args.aa_samples}")

    train_loader, test_loader, testset = get_loaders(args.batch)

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

    models = {}
    results = {"seed": args.seed, "epochs": args.epochs,
               "eps": args.eps, "aa_samples": args.aa_samples,
               "distill_T": args.distill_t, "device": device,
               "models": {}}

    progress = Progress(os.path.join(REPORT_DIR, "progress.json"))
    progress.set(phase="training", current_model="(初始化)")

    # ---------- 训练 / 加载各模型 ----------
    spec = [
        ("baseline",  {"lr": args.lr, "T": 1.0, "teacher": None}),
        ("mask_sat",  {"lr": args.lr, "T": 1.0, "teacher": None}),
        ("mask_prep", {"lr": args.lr, "T": 1.0, "teacher": None}),
    ]
    for kind, cfg in spec:
        ckpt = os.path.join(CKPT_DIR, f"{kind}.pth")
        m = make_model(kind, args.seed).to(device)
        progress.set(models={**progress.data["models"], kind: {}})
        if args.reuse and os.path.exists(ckpt):
            m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            print(f"[load] {kind} <- {ckpt}")
            progress.log(f"加载 {kind} checkpoint")
        else:
            print(f"[train] {kind} ...")
            progress.log(f"开始训练 {kind}")
            t0 = time.time()
            train_model(m, train_loader, test_loader, device,
                        args.epochs, cfg["lr"], T=cfg["T"],
                        teacher=cfg["teacher"], tag=kind, progress=progress)
            torch.save(m.state_dict(), ckpt)
            print(f"[train] {kind} done in {time.time()-t0:.0f}s -> {ckpt}")
            progress.log(f"{kind} 训练完成 ({time.time()-t0:.0f}s)")
        models[kind] = m
        results["models"][kind] = {"clean": None}

    # 防御蒸馏 (基于调研的标准做法: 教师 T=1 正常训练 + 学生高温软标签训练)
    teacher_ckpt = os.path.join(CKPT_DIR, "teacher.pth")
    if args.reuse and os.path.exists(teacher_ckpt):
        teacher = NormalizeNet(LeNet()).to(device)
        teacher.load_state_dict(torch.load(teacher_ckpt, map_location=device,
                                           weights_only=True))
        print(f"[load] teacher <- {teacher_ckpt}")
    else:
        teacher = NormalizeNet(LeNet()).to(device)
        print("[train] teacher (T=1 正常训练, 供软标签) ...")
        progress.log("开始训练教师网络 (T=1)")
        t0 = time.time()
        train_model(teacher, train_loader, test_loader, device,
                    args.epochs, args.lr, T=1.0,
                    teacher=None, tag="teacher", progress=progress)
        torch.save(teacher.state_dict(), teacher_ckpt)
        print(f"[train] teacher done in {time.time()-t0:.0f}s")
        progress.log(f"教师网络训练完成 ({time.time()-t0:.0f}s)")
    models["teacher"] = teacher
    results["models"]["teacher"] = {"clean": None}

    student_ckpt = os.path.join(CKPT_DIR, "distill.pth")
    if args.reuse and os.path.exists(student_ckpt):
        student = NormalizeNet(LeNet()).to(device)
        student.load_state_dict(torch.load(student_ckpt, map_location=device,
                                           weights_only=True))
        print(f"[load] distill <- {student_ckpt}")
    else:
        student = NormalizeNet(LeNet()).to(device)
        print(f"[train] distill (student, T={args.distill_t}, soft labels, "
              f"lr={args.student_lr}, wd=0) ...")
        progress.log(f"开始训练蒸馏学生网络 (T={args.distill_t})")
        t0 = time.time()
        train_model(student, train_loader, test_loader, device,
                    args.epochs, args.student_lr, T=args.distill_t,
                    teacher=teacher, tag="distill", wd=0.0, progress=progress)
        torch.save(student.state_dict(), student_ckpt)
        print(f"[train] distill done in {time.time()-t0:.0f}s")
        progress.log(f"蒸馏学生网络训练完成 ({time.time()-t0:.0f}s)")
    models["distill"] = student
    results["models"]["distill"] = {"clean": None}

    # ---------- 干净准确率 ----------
    print("\n========== 1) 干净测试准确率 ==========")
    progress.set(phase="eval_clean", current_model="(评估干净准确率)", eta_s=None)
    for kind, m in models.items():
        results["models"][kind]["clean"] = round(
            evaluate(m, test_loader, device, desc=kind), 2)
        progress.set(models={**progress.data["models"], kind:
                             {"clean": results["models"][kind]["clean"]}})

    # ---------- 2) AutoAttack ----------
    if not args.no_aa:
        print("\n========== 2) AutoAttack (Linf, eps=8/255) ==========")
        progress.set(phase="attacking", current_attack="AutoAttack")
        for kind in ("baseline", "distill", "mask_sat", "mask_prep"):
            print(f"\n  --- AA 攻击 {kind} ---")
            progress.log(f"AutoAttack 攻击 {kind} ...")
            rob, dt = run_autoattack(models[kind], x_attack, y_attack, eps,
                                     args.aa_batch, args.aa_square_queries,
                                     args.aa_iters, device)
            results["models"][kind]["aa_robust"] = round(float(rob), 2)
            results["models"][kind]["aa_time_s"] = round(dt, 1)
            print(f"  [AA] {kind} robust acc = {rob:.2f}%  ({dt:.0f}s)")
            progress.set(models={**progress.data["models"], kind:
                                 {**progress.data["models"].get(kind, {}),
                                  "aa_robust": results["models"][kind]["aa_robust"]}})
            progress.log(f"AutoAttack {kind} 完成: 鲁棒率 {rob:.2f}%")

    # ---------- 3) PGD-20 白盒 ----------
    print("\n========== 3) PGD-20 白盒攻击 ==========")
    progress.set(phase="attacking", current_attack="PGD-20")
    for kind in ("baseline", "distill", "mask_sat", "mask_prep"):
        acc = attack_eval(models[kind], x_attack, y_attack, eps, alpha,
                          args.pgd_iters, device=device)
        results["models"][kind]["pgd20"] = round(acc, 2)
        print(f"  [PGD] {kind}: robust acc = {acc:.2f}%")
        progress.set(models={**progress.data["models"], kind:
                             {**progress.data["models"].get(kind, {}),
                              "pgd20": results["models"][kind]["pgd20"]}})

    # ---------- 4) BPDA-PGD-20 (拆穿梯度遮蔽) ----------
    print("\n========== 4) BPDA-PGD-20 (自适应攻击) ==========")
    progress.set(phase="attacking", current_attack="BPDA-PGD-20")
    surrogate_sat = NormalizeNet(models["mask_sat"].net).to(device)   # 跳过 log_sigmoid
    surrogate_prep = NormalizeNet(models["mask_prep"].net).to(device) # 跳过位深量化
    for kind, sur in (("mask_sat", surrogate_sat), ("mask_prep", surrogate_prep)):
        acc = attack_eval(models[kind], x_attack, y_attack, eps, alpha,
                          args.pgd_iters, surrogate=sur, device=device)
        results["models"][kind]["bpda20"] = round(acc, 2)
        print(f"  [BPDA] {kind}: robust acc = {acc:.2f}%")
        progress.set(models={**progress.data["models"], kind:
                             {**progress.data["models"].get(kind, {}),
                              "bpda20": results["models"][kind]["bpda20"]}})

    # ---------- 5) 迁移攻击: baseline 生成的 AE 打到各防御模型 ----------
    print("\n========== 5) 迁移攻击 (baseline 的 PGD-20 对抗样本) ==========")
    progress.set(phase="attacking", current_attack="迁移攻击")
    x_adv_base = pgd_attack(models["baseline"], x_attack, y_attack, eps, alpha,
                            args.pgd_iters, device=device)
    for kind in ("baseline", "distill", "mask_sat", "mask_prep"):
        with torch.no_grad():
            acc = (models[kind](x_adv_base).argmax(1) == y_attack).float().mean().item() * 100
        results["models"][kind]["transfer20"] = round(acc, 2)
        print(f"  [transfer] {kind}: robust acc = {acc:.2f}%")
        progress.set(models={**progress.data["models"], kind:
                             {**progress.data["models"].get(kind, {}),
                              "transfer20": results["models"][kind]["transfer20"]}})

    # ---------- 汇总 ----------
    print("\n========== 实验汇总 (robust acc = 越小越不安全) ==========")
    hdr = f"{'模型':<10}{'clean%':>9}{'AA%':>9}{'PGD20%':>9}{'BPDA%':>9}{'transfer%':>11}"
    print(hdr)
    print("-" * len(hdr))
    for kind in ("baseline", "distill", "mask_sat", "mask_prep"):
        r = results["models"][kind]
        print(f"{kind:<10}{r['clean']:>9.2f}"
              f"{r.get('aa_robust', float('nan')):>9.2f}"
              f"{r['pgd20']:>9.2f}"
              f"{r.get('bpda20', float('nan')):>9.2f}"
              f"{r['transfer20']:>11.2f}")

    report_path = os.path.join(REPORT_DIR, "defense_experiment.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[report] 已保存 -> {report_path}")

    progress.set(status="done", phase="done", current_model="(完成)",
                 current_epoch=args.epochs, eta_s=None,
                 models={**progress.data["models"],
                         **{k: {**v, "clean": results["models"][k]["clean"]}
                            for k, v in progress.data["models"].items()}})
    progress.log("实验完成")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()
        # 出错时把错误状态写进 progress.json, 让监控页面显示出来
        try:
            pp = os.path.join(REPORT_DIR, "progress.json")
            with open(pp, encoding="utf-8") as f:
                data = json.load(f)
            data["status"] = "error"
            data["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            data["error"] = traceback.format_exc()[-1500:]
            with open(pp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception:
            pass
        raise
