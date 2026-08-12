import os
import sys
# 将父目录加入 Python 路径，使 transferattack 包可被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 解决 Windows 上 torch 与 anaconda 的 OpenMP 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import tqdm
from PIL import Image
import transferattack
from transferattack.utils import (save_images, load_pretrained_model,
    wrap_model, cnn_model_paper, vit_model_paper,
    generation_target_classes, AdvDataset)


# 自定义数据集类，使用 CIFAR-10 数据
class CIFARDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir, output_dir, targeted=False, eval=False):
        full_dataset = torchvision.datasets.CIFAR10(
            root='../data',
            train=False,
            download=False  # 你已经有了数据，不需要下载
        )
        # 只取前 50 张图片
        self.dataset = torch.utils.data.Subset(full_dataset, range(50))
        print(f"加载了 {len(self.dataset)} 张 CIFAR-10 图片")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if img.mode != 'RGB':
            img = img.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img)
        filename = f"cifar_{idx:05d}.png"
        return img_tensor, label, filename


def denormalize(images):
    """反标准化：将标准化后的 tensor 恢复成正常颜色范围 [0,1]"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    return images


def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--attack', default='mifgsm', type=str, help='the attack algorithm',
                        choices=transferattack.attack_zoo.keys())
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--batchsize', default=32, type=int, help='the bacth size')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--model', default='resnet50', type=str, help='the source surrogate model')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='./data', type=str,
                        help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--output_dir', default='./results', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='0', type=str)
    parser.add_argument('--nat_attacked_neuron', default=250, type=int,
                        help='NAT attack: target neuron index (default: 250)')
    parser.add_argument('--save_original', action='store_true', help='also save original images')
    return parser.parse_args()


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = CIFARDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=0)

    if not args.eval:
        if args.ensemble or len(args.model.split(',')) > 1:
            args.model = args.model.split(',')

        # Build kwargs for attack-specific parameters
        attack_kwargs = {'model_name': args.model, 'targeted': args.targeted}
        if args.attack == 'nat':
            attack_kwargs['nat_attacked_neuron'] = args.nat_attacked_neuron
        attack_kwargs['device'] = torch.device(f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu')
        attacker = transferattack.load_attack_class(args.attack)(**attack_kwargs)

        for batch_idx, [images, labels, filenames] in tqdm.tqdm(enumerate(dataloader)):
            # 保存原始图片（如果启用）- 修复颜色
            if args.save_original:
                original_dir = os.path.join(args.output_dir, 'original')
                if not os.path.exists(original_dir):
                    os.makedirs(original_dir)
                # 反标准化后再保存
                original_images_denorm = denormalize(images)
                save_images(original_dir, original_images_denorm.cpu(), filenames)

            if args.attack in ['ttp', 'm3d', 'rfcoa', 'aim']:
                for idx, target_class in enumerate(generation_target_classes):
                    perturbations = attacker(images, labels, idx)
                    new_output_dir = os.path.join(args.output_dir, str(target_class))
                    if not os.path.exists(new_output_dir):
                        os.makedirs(new_output_dir)
                    # 对抗样本也要修复颜色
                    adv_images_denorm = denormalize(images + perturbations)
                    save_images(new_output_dir, adv_images_denorm.cpu(), filenames)
            else:
                perturbations = attacker(images, labels)
                # 对抗样本也要修复颜色
                adv_images_denorm = denormalize(images + perturbations)
                save_images(args.output_dir, adv_images_denorm.cpu(), filenames)
    else:
        res = '|'
        for model_name, model in load_pretrained_model(cnn_model_paper, vit_model_paper):
            model = wrap_model(model.eval())
            for p in model.parameters():
                p.requires_grad = False

            if args.attack in ['ttp', 'm3d', 'rfcoa']:
                asr = 0
                for idx, target_class in enumerate(generation_target_classes):
                    new_output_dir = os.path.join(args.output_dir, str(target_class))
                    new_dataset = AdvDataset(input_dir=args.input_dir, output_dir=new_output_dir, targeted=True,
                                             target_class=target_class, eval=args.eval)
                    new_dataloader = torch.utils.data.DataLoader(new_dataset, batch_size=args.batchsize, shuffle=False,
                                                                 num_workers=0)
                    asr += eval(model, new_dataloader, True)
                asr /= 10
            else:
                asr = eval(model, dataloader, args.targeted)
            print(f'{model_name}: {asr:.1f}')
            res += f' {asr:.1f} |'

        print(res)
        with open('results_eval.txt', 'a') as f:
            f.write(args.output_dir + res + '\n')


def eval(model, dataloader, is_targeted):
    correct, total = 0, 0
    for images, labels, _ in dataloader:
        if is_targeted:
            labels = labels[1]
        pred = model(images)
        correct += (labels.cpu().numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
        total += labels.shape[0]
    if is_targeted:
        # correct: pred == target_label
        asr = (correct / total) * 100
    else:
        # correct: pred == original_label
        asr = (1 - correct / total) * 100
    return asr


if __name__ == '__main__':
    main()