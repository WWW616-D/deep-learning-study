"""模型清单核对: 逐个加载 checkpoint, 报告架构/参数量/干净准确率 (实测, 不靠 JSON)."""
import os, sys, importlib, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\py\code")
T = importlib.import_module("try")
import adv_train as A
import pixeldp_train as P
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

device = "cpu"
N_EVAL, DRAWS = 2000, 20      # 干净准确率用 2000 张 + 20 次抽样投票 (随机化模型)

_, _, testset = T.get_loaders(128)
rng = np.random.RandomState(123)
idx = rng.choice(len(testset), size=N_EVAL, replace=False)
x = torch.stack([testset[i][0] for i in idx]).to(device)
y = torch.tensor([testset[i][1] for i in idx]).to(device)

CKPT = T.CKPT_DIR
rows = []
for fn in sorted(os.listdir(CKPT)):
    if not fn.endswith(".pth"):
        continue
    sd = torch.load(os.path.join(CKPT, fn), map_location=device, weights_only=True)
    keys = list(sd.keys())
    n_param = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    # 判定网络类型
    if any(k.startswith("base.net.") for k in keys):
        # PixelDPNet: base.net.* -> 取 sigma 无法从权重得知, 用文件名推断
        sigma = {"pixeldp_L0.03_c1.pth": 0.0761, "pixeldp_L0.1_c1.pth": 0.2537,
                 "pixeldp_L0.2_c1.pth": 0.5075}.get(fn, None)
        m = P.PixelDPNet(T.NormalizeNet(A.C1Net()), sigma or 0.25, "gaussian").to(device)
        m.load_state_dict(sd); m.eval()
        with torch.no_grad():
            votes = None
            for _ in range(DRAWS):
                lab = m(x).argmax(1)
                oh = F.one_hot(lab, 10).float()
                votes = oh if votes is None else votes + oh
            acc = float((votes.argmax(1) == y).float().mean() * 100)
        arch = f"C1Net(200K) + 输入高斯噪声 σ={sigma}"
        n_net = sum(v.numel() for k, v in sd.items() if k.startswith("base.net."))
        rows.append((fn, arch, n_net, acc, "20次投票"))
    elif "kernel" in keys:
        # DiffuseNet: keys = net.conv1.weight.../kernel (内层直接是 C1Net, 归一化写在 forward)
        inner = {k[len("net."):]: v for k, v in sd.items() if k.startswith("net.")}
        raw = A.C1Net().to(device)
        raw.load_state_dict(inner); raw.eval()
        base = T.NormalizeNet(raw).to(device); base.eval()   # 等价于 DiffuseNet 的归一化
        n_net = sum(v.numel() for v in inner.values())
        arch = "C1Net(200K) + 输入扩散变换"
        with torch.no_grad():
            acc = float((base(x).argmax(1) == y).float().mean() * 100)
        rows.append((fn, arch, n_net, acc, "单次前向"))
    elif any(k.startswith("net.") for k in keys):
        n_net = sum(v.numel() for k, v in sd.items() if k.startswith("net."))
        # mask_sat / mask_prep 的 forward 带变换 (且 SaturatedNet/PreprocessNet 不做归一化),
        # 必须用各自的 wrapper 类评估, 否则数字会错
        if fn == "mask_sat.pth":
            base = T.SaturatedNet(T.LeNet(), gain=1.0).to(device)
            arch = "LeNet (62K) + log_sigmoid 饱和输出"
        elif fn == "mask_prep.pth":
            base = T.PreprocessNet(T.LeNet(), bits=5).to(device)
            arch = "LeNet (62K) + 5bit 位深缩减"
        elif n_net > 150000:            # C1Net
            base = T.NormalizeNet(A.C1Net()).to(device)
            arch = "C1Net (200K)"
        else:                          # LeNet
            base = T.NormalizeNet(T.LeNet()).to(device)
            arch = "LeNet (62K)"
        base.load_state_dict(sd); base.eval()
        with torch.no_grad():
            acc = float((base(x).argmax(1) == y).float().mean() * 100)
        rows.append((fn, arch, n_net, acc, "单次前向"))
    else:
        rows.append((fn, "未知结构", n_param, None, "-"))

print(f"{'文件':<24}{'架构':<38}{'参数量':>9}  {'干净准确率':>10}  备注")
print("-" * 100)
for fn, arch, n_net, acc, note in rows:
    a = f"{acc:.2f}%" if acc is not None else "—"
    print(f"{fn:<24}{arch:<38}{n_net:>9,}  {a:>10}  {note}")

out = [{"file": r[0], "arch": r[1], "params": r[2],
        "clean_acc_subset2000": round(r[3], 2) if r[3] is not None else None,
        "note": r[4]} for r in rows]
rp = os.path.join(T.REPORT_DIR, "model_inventory.json")
with open(rp, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n[report] -> {rp}")
