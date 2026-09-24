"""诊断对抗损失的量级, 判断是否梯度爆炸. 打印干净损失 vs 对抗损失, 及梯度范数."""
import os, sys, importlib
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, r"D:\py\code")
T = importlib.import_module("try")
import torch
import torch.nn.functional as F

model = T.make_model("baseline", 0)
model.load_state_dict(torch.load(os.path.join(T.CKPT_DIR, "baseline.pth"),
                                 map_location="cpu", weights_only=True))
train_loader, _, _ = T.get_loaders(128)
eps, alpha = 8/255.0, 2/255.0
model.eval()
x, y = next(iter(train_loader))
# 干净损失
loss_clean = F.cross_entropy(model(x), y)
# 对抗损失
x0 = x.clone().detach()
delta = torch.zeros_like(x0).uniform_(-eps, eps).clamp(-x0, 1-x0)
delta.requires_grad_(True)
for _ in range(7):
    xa = torch.clamp(x0 + delta, 0, 1)
    l = F.cross_entropy(model(xa), y)
    g = torch.autograd.grad(l, delta)[0]
    delta = (delta + alpha*g.sign()).clamp(-eps, eps).clamp(-x0, 1-x0)
x_adv = torch.clamp(x0 + delta.detach(), 0, 1)
loss_adv = F.cross_entropy(model(x_adv), y)
print(f"干净损失 = {loss_clean.item():.4f}")
print(f"对抗损失 = {loss_adv.item():.4f}")
print(f"对抗/干净 = {loss_adv.item()/loss_clean.item():.2f}x")
# 对抗样本的预测分布
p = model(x_adv).argmax(1)
print(f"对抗样本上模型正确率: {(p==y).float().mean().item()*100:.1f}%")
# 干净样本在对抗训练前的预测 (应接近 71%)
p2 = model(x).argmax(1)
print(f"干净样本上模型正确率: {(p2==y).float().mean().item()*100:.1f}%")
# 干净梯度范数 vs 对抗梯度范数
loss_clean.backward(retain_graph=False)
g_clean = sum(p_.grad.norm().item() for p_ in model.parameters() if p_.grad is not None)
model.zero_grad()
# 对抗梯度 (单独算)
l2 = F.cross_entropy(model(x_adv), y)
l2.backward()
g_adv = sum(p_.grad.norm().item() for p_ in model.parameters() if p_.grad is not None)
print(f"干净梯度范数和 = {g_clean:.3f}")
print(f"对抗梯度范数和 = {g_adv:.3f}")
print(f"梯度比 = {g_adv/g_clean:.2f}x")
