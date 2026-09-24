"""诊断: 温度 T 与权重衰减对梯度信号的影响 (10 个 batch 快速对照)"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

torch.manual_seed(0)

def make_net():
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 6, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(6, 16, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(),
        torch.nn.Linear(16*5*5, 120), torch.nn.ReLU(),
        torch.nn.Linear(120, 84), torch.nn.ReLU(),
        torch.nn.Linear(84, 10))

trainset = torchvision.datasets.CIFAR10(
    root=r"D:\py\data", train=True, download=False,
    transform=T.Compose([T.ToTensor()]))
loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
it = iter(loader)

def run(tag, T, lr, wd, n_batches=10):
    net = make_net()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    print(f"\n=== {tag}: T={T} lr={lr} wd={wd} ===")
    for b in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            it2 = iter(loader); x, y = next(it2)
        opt.zero_grad()
        z = net(x)
        loss = F.cross_entropy(z / T, y)
        loss.backward()
        opt.step()
        if b in (0, 4, 9):
            w3 = net[-1].weight.detach()
            g3 = net[-1].weight.grad
            zt = z.gather(1, y.view(-1, 1)).detach().mean().item()
            zo = z.detach().mean().item()
            print(f"  batch{b:2d} loss={loss.item():.4f} |fc3.w|={w3.norm().item():.4f} "
                  f"|fc3.grad|={g3.norm().item():.5f} z_target={zt:+.3f} z_other={zo:+.3f}")

run("T=1  正常", 1, 0.1, 5e-4)
run("T=100 wd 开", 100, 1.0, 5e-4)
run("T=100 wd 关", 100, 1.0, 0.0)
run("T=100 wd 开 lr=10", 100, 10.0, 5e-4)
