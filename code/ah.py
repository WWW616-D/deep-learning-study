import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchattacks
import urllib.request
import json

# ==================== 1. 准备工作 ====================
print("=" * 50)
print("使用 Torchattacks 实现 MI-FGSM")
print("=" * 50)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n设备: {device}")

# 加载预训练模型
print("\n[1/3] 加载模型...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model = model.to(device)
model.eval()
print("模型: ResNet50 (ImageNet 预训练)")

# 加载 ImageNet 类别标签（使用 PyTorch 内置方式）
print("\n[2/3] 加载标签...")
try:
    # 方法1：直接从 PyTorch hub 加载
    import requests
    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    response = requests.get(url, timeout=5)
    imagenet_labels = [line.strip() for line in response.text.splitlines()]
    print("标签加载成功（网络）")
except:
    # 方法2：如果网络不行，手动创建常用标签（前100个）
    print("网络加载失败，使用本地内置标签...")
    # 这里只列出部分常见类别，足够演示使用
    imagenet_labels = [
        'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
        'electric ray', 'stingray', 'cock', 'hen', 'ostrich',
        'brambling', 'goldfinch', 'house finch', 'junco', 'indigo bunting',
        'robin', 'bulbul', 'jay', 'magpie', 'chickadee',
        'water ouzel', 'kite', 'bald eagle', 'vulture', 'great grey owl',
        'European fire salamander', 'common newt', 'eft', 'spotted salamander', 'axolotl',
        'bullfrog', 'tree frog', 'tailed frog', 'loggerhead', 'leatherback turtle',
        'mud turtle', 'terrapin', 'box turtle', 'banded gecko', 'common iguana',
        'American chameleon', 'whiptail', 'agama', 'frilled lizard', 'alligator lizard',
        'Gila monster', 'green lizard', 'African chameleon', 'Komodo dragon', 'African crocodile',
        'American alligator', 'triceratops', 'thunder snake', 'ringneck snake', 'hognose snake',
        'green snake', 'king snake', 'garter snake', 'water snake', 'vine snake',
        'night snake', 'boa constrictor', 'rock python', 'Indian cobra', 'green mamba',
        'sea snake', 'horned viper', 'diamondback', 'sidewinder', 'trilobite',
        'harvestman', 'scorpion', 'black and gold garden spider', 'barn spider', 'garden spider',
        'black widow', 'tarantula', 'wolf spider', 'tick', 'centipede',
        'black grouse', 'ptarmigan', 'ruffed grouse', 'prairie chicken', 'peacock',
        'quail', 'partridge', 'African grey', 'macaw', 'sulphur-crested cockatoo',
        'lorikeet', 'coucal', 'bee eater', 'hornbill', 'hummingbird',
        'jacamar', 'toucan', 'drake', 'red-breasted merganser', 'goose',
        'black swan', 'tusker', 'echidna', 'platypus', 'wallaby',
        # ... 实际有1000个类别，这里只列出前100个作为示例
    ]
    # 补全到1000个（用占位符）
    while len(imagenet_labels) < 1000:
        imagenet_labels.append(f'class_{len(imagenet_labels)}')
    print(f"使用本地标签（共{len(imagenet_labels)}个）")

image_path = "../data/test_dog.jpg"  # 如果图片在代码同目录就叫这个名字
# image_path = "C:/Users/17944/Desktop/dog.jpg"  # 或者写完整路径

try:
    image = Image.open(image_path).convert('RGB')
    print(f"图片加载成功: {image_path}")
    print(f"图片尺寸: {image.size}")
except FileNotFoundError:
    print(f"错误：找不到图片 '{image_path}'")
    print("请确保图片路径正确，或将图片放到代码同目录并命名为 'test_dog.jpg'")
    exit()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img_tensor = transform(image).unsqueeze(0).to(device)

# 获取原始预测
with torch.no_grad():
    out = model(img_tensor)
    orig_idx = out.argmax(dim=1).item()
    orig_conf = torch.softmax(out, dim=1).max().item()

print(f"\n原始图像预测:")
print(f"  类别索引: {orig_idx}")
if orig_idx < len(imagenet_labels):
    print(f"  类别名称: {imagenet_labels[orig_idx]}")
print(f"  置信度: {orig_conf:.2%}")

# ==================== 2. 使用 torchattacks 执行 MI-FGSM ====================
print("\n" + "=" * 50)
print("执行 MI-FGSM 攻击")
print("=" * 50)

# 创建 MI-FGSM 攻击器
atk = torchattacks.MIFGSM(
    model,
    eps=16/255,      # 最大扰动幅度
    steps=10,        # 迭代次数
    decay=1.0        # 动量衰减因子
)

# 生成对抗样本
label_tensor = torch.tensor([orig_idx]).to(device)
adv_tensor = atk(img_tensor, label_tensor)

print("攻击完成！")
print(f"参数: eps={16/255:.4f}, steps=10, decay=1.0")

# 获取对抗样本预测
with torch.no_grad():
    adv_out = model(adv_tensor)
    adv_idx = adv_out.argmax(dim=1).item()
    adv_conf = torch.softmax(adv_out, dim=1).max().item()

print(f"\n对抗图像预测:")
print(f"  类别索引: {adv_idx}")
if adv_idx < len(imagenet_labels):
    print(f"  类别名称: {imagenet_labels[adv_idx]}")
print(f"  置信度: {adv_conf:.2%}")

# ==================== 3. 可视化 ====================
print("\n" + "=" * 50)
print("生成可视化结果")
print("=" * 50)

# 转换为 numpy
orig_np = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
adv_np = adv_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
noise = adv_np - orig_np
noise_amp = noise * 10 + 0.5  # 放大 10 倍
noise_amp = np.clip(noise_amp, 0, 1)

# 绘图
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# 原始图像标题
orig_title = f'Original\nClass {orig_idx}'
if orig_idx < len(imagenet_labels):
    orig_title = f'Original\n{imagenet_labels[orig_idx]}'
axes[0].set_title(f'{orig_title} ({orig_conf:.2%})', fontsize=12)
axes[0].imshow(orig_np)
axes[0].axis('off')

# 噪声
axes[1].imshow(noise_amp)
axes[1].set_title('Perturbation (10x amplified)', fontsize=12)
axes[1].axis('off')

# 对抗图像标题
adv_title = f'Adversarial\nClass {adv_idx}'
if adv_idx < len(imagenet_labels):
    adv_title = f'Adversarial\n{imagenet_labels[adv_idx]}'
color = 'red' if adv_idx != orig_idx else 'green'
axes[2].set_title(f'{adv_title} ({adv_conf:.2%})', fontsize=12, color=color)
axes[2].imshow(adv_np)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('mifgsm_torchattacks_result.png', dpi=150, bbox_inches='tight')
plt.show()

# 结果判定
print("\n" + "=" * 50)
if adv_idx != orig_idx:
    print("✅ 攻击成功！模型被欺骗。")
else:
    print("❌ 攻击失败，尝试增加 eps 或 steps。")
print("=" * 50)