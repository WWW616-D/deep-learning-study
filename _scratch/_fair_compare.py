"""公平对比: 对已有的对抗训练模型(adv_train_c1)跑与 PixelDP 相同的 AA-rand(EOT) 协议."""
import os, sys, importlib, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\py\code")
T = importlib.import_module("try")
import adv_train as A
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

device = "cpu"
eps = 8/255.0
aa_samples, aa_batch, eot = 500, 100, 10

model = T.NormalizeNet(A.C1Net()).to(device)
model.load_state_dict(torch.load(os.path.join(T.CKPT_DIR, "adv_train_c1.pth"),
                                 map_location=device, weights_only=True))
print("[load] adv_train_c1")

_, _, testset = T.get_loaders(128)
rng = np.random.RandomState(0)
idx = rng.choice(len(testset), size=aa_samples, replace=False)
x_aa = torch.stack([testset[i][0] for i in idx]).to(device)
y_aa = torch.tensor([testset[i][1] for i in idx]).to(device)

aa_src = os.path.join(T.ROOT, "auto-attack-master")
sys.path.insert(0, aa_src)
from autoattack import AutoAttack

model.eval()
adv = AutoAttack(model, norm='Linf', eps=eps, version='rand',
                 device=device, verbose=False)
if hasattr(adv.apgd, "eot_iter"):
    adv.apgd.eot_iter = eot
t0 = time.time()
x_adv = adv.run_standard_evaluation(x_aa, y_aa, bs=aa_batch)
with torch.no_grad():
    rob = (model(x_adv).argmax(1) == y_aa).float().mean().item() * 100
print(f"[AA-rand] adv_train_c1 鲁棒率 = {rob:.2f}%  ({time.time()-t0:.0f}s)")

# 同时用 standard 协议跑一遍做交叉参照
adv2 = AutoAttack(model, norm='Linf', eps=eps, version='standard',
                  device=device, verbose=False)
t0 = time.time()
x_adv2 = adv2.run_standard_evaluation(x_aa, y_aa, bs=aa_batch)
with torch.no_grad():
    rob2 = (model(x_adv2).argmax(1) == y_aa).float().mean().item() * 100
print(f"[AA-standard] adv_train_c1 鲁棒率 = {rob2:.2f}%  ({time.time()-t0:.0f}s)")
