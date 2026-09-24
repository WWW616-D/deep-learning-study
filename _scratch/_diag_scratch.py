"""验证: 从头做对抗训练 (随机初始化 + PGD-7) 是否 clean 能正常爬升 (而非崩到 10%)."""
import os, sys, importlib
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\py\code")
T = importlib.import_module("try")
import torch
import torch.nn.functional as F

def adv_train_scratch(lr, pgd_iters, tag):
    torch.manual_seed(0)
    model = T.make_model("baseline", 0)
    train_loader, test_loader, _ = T.get_loaders(128)
    eps, alpha = 8/255.0, 2/255.0
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    for ep in range(1, 6):
        model.train()
        for x, y in train_loader:
            model.eval()
            x0 = x.clone().detach()
            delta = torch.zeros_like(x0).uniform_(-eps, eps).clamp(-x0, 1-x0)
            delta.requires_grad_(True)
            for _ in range(pgd_iters):
                xa = torch.clamp(x0 + delta, 0, 1)
                l = F.cross_entropy(model(xa), y)
                g = torch.autograd.grad(l, delta)[0]
                delta = (delta + alpha*g.sign()).clamp(-eps, eps).clamp(-x0, 1-x0)
            x_adv = torch.clamp(x0 + delta.detach(), 0, 1)
            model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(x_adv), y)
            loss.backward(); opt.step()
        sched.step()
        acc = T.evaluate(model, test_loader, "cpu", desc=f"{tag} lr{lr} ep{ep}")
    return acc

print("=== 从头对抗训练 (随机初始化 + PGD-7, 5 epoch) ===")
adv_train_scratch(0.1, 7, "scratch-PGD7")
