"""补充检查: 对 PixelDP 模型跑 standard AA (含免梯度 Square), 看梯度无关攻击能否击穿."""
import os, sys, importlib, time
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\py\code")
T = importlib.import_module("try")
import adv_train as A
import pixeldp_train as P
import numpy as np
import torch

device = "cpu"
eps = 8/255.0
delta, mech = 0.05, "gaussian"

_, _, testset = T.get_loaders(128)
rng = np.random.RandomState(0)
idx = rng.choice(len(testset), size=500, replace=False)
x_aa = torch.stack([testset[i][0] for i in idx]).to(device)
y_aa = torch.tensor([testset[i][1] for i in idx]).to(device)

sys.path.insert(0, os.path.join(T.ROOT, "auto-attack-master"))
from autoattack import AutoAttack

for L in (0.1, 0.03):
    sigma = P.sigma_from_L(L, 1.0, delta, mech)
    model = P.PixelDPNet(T.NormalizeNet(A.C1Net()), sigma, mech).to(device)
    model.load_state_dict(torch.load(os.path.join(T.CKPT_DIR, f"pixeldp_L{L}_c1.pth"),
                                     map_location=device, weights_only=True))
    model.eval()
    adv = AutoAttack(model, norm='Linf', eps=eps, version='standard',
                     device=device, verbose=False)
    if hasattr(adv.square, "n_queries"):
        adv.square.n_queries = 1500
    t0 = time.time()
    x_adv = adv.run_standard_evaluation(x_aa, y_aa, bs=100)
    with torch.no_grad():
        accs = [(model(x_adv).argmax(1) == y_aa).float().mean().item() for _ in range(5)]
    print(f"[AA-standard] pixeldp L={L} 鲁棒率 = {np.mean(accs)*100:.2f}%  ({time.time()-t0:.0f}s)")
