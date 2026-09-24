"""诊断: 对抗训练 clean 掉到 11% 的原因. 对照不同 lr 从 baseline 微调 2 epoch."""
import os, sys, importlib
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\py\code")
T = importlib.import_module("try")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

CKPT = os.path.join(T.CKPT_DIR, "baseline.pth")

def run(lr, pgd_iters=7, tag=""):
    torch.manual_seed(0)
    model = T.make_model("baseline", 0)
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    train_loader, test_loader, _ = T.get_loaders(128)
    eps, alpha = 8/255.0, 2/255.0
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for ep in range(1, 3):
        model.train()
        total_loss = n = 0
        for x, y in train_loader:
            # PGD-7 生成对抗样本
            model.eval()
            x0 = x.clone().detach()
            delta = torch.zeros_like(x0).uniform_(-eps, eps).clamp(-x0, 1-x0)
            delta.requires_grad_(True)
            for _ in range(pgd_iters):
                xa = torch.clamp(x0 + delta, 0, 1)
                loss = F.cross_entropy(model(xa), y)
                g = torch.autograd.grad(loss, delta)[0]
                delta = (delta + alpha * g.sign()).clamp(-eps, eps).clamp(-x0, 1-x0)
            x_adv = torch.clamp(x0 + delta.detach(), 0, 1)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()*x.size(0); n += x.size(0)
        acc = T.evaluate(model, test_loader, "cpu", desc=f"{tag} lr{lr} ep{ep}")
    return acc

print("=== 对照组: 不同 lr 从 baseline 微调 2 epoch ===")
run(0.01, 7, "PGD7")
run(0.001, 7, "PGD7")
run(0.0001, 7, "PGD7")
