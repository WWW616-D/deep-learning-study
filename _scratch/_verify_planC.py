"""验证方案 C (Papernot/Carlini 原文配方) 在我们的 CIFAR+LeNet 上是否可训练
配方: T=100, wd=0, lr=0.01*(0.5**int(epoch/10)), momentum=0.9*(0.5**int(epoch/10))
对照组: T=100, wd=0, lr=1.0 (固定)
"""
import os, sys, importlib
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\py\code")
t = importlib.import_module("try")
NormalizeNet, LeNet = t.NormalizeNet, t.LeNet
get_loaders, evaluate = t.get_loaders, t.evaluate
import torch
import torch.nn.functional as F

device = "cpu"
train_loader, test_loader, _ = get_loaders(128)

def run_planC(tag, base_lr, epochs=20, T=100.0):
    torch.manual_seed(0)
    net = NormalizeNet(LeNet())
    # 论文配方: lr 与 momentum 每 10 epoch 减半
    opt = torch.optim.SGD(net.parameters(), lr=base_lr, momentum=0.9)  # wd=0 !
    print(f"\n=== {tag}: T={T} base_lr={base_lr} wd=0 ===")
    step = 0
    for ep in range(1, epochs + 1):
        net.train()
        for x, y in train_loader:
            opt.zero_grad()
            loss = F.cross_entropy(net(x) / T, y)
            loss.backward()
            opt.step()
            step += 1
        # 手动 lr/momentum 衰减 (每10 epoch 减半), 与原论文一致
        decay = 0.5 ** (ep // 10)
        for g in opt.param_groups:
            g['lr'] = base_lr * decay
            g['momentum'] = 0.9 * decay
        if ep in (1, 5, 10, 15, 20):
            acc = evaluate(net, test_loader, device, desc=f"{tag} ep{ep}")
    return net

net_a = run_planC("Carlini配方 lr0.01", 0.01)
net_b = run_planC("对照组 lr1.0", 1.0)
print("\n[结论] 打印最终准确率:",
      evaluate(net_a, test_loader, device, desc="A lr0.01 final"),
      evaluate(net_b, test_loader, device, desc="B lr1.0 final"))
