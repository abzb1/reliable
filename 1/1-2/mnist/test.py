import os
import random

import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from model import MLP300K
from utils import mnist_train_model

# Fix random seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# MNIST dataset
transform_pipeline = ToTensor()
train_dataset = MNIST(root='data/', train=True, transform=transform_pipeline, download=True)
test_dataset = MNIST(root='data/', train=False, transform=transform_pipeline)

def fgsm_untargeted(model, x, label, eps):
    x_adv = x.clone().detach().requires_grad_(True)
    outputs = model(x_adv)
    loss = F.cross_entropy(outputs, label)
    model.zero_grad()
    loss.backward()

    x_adv = x_adv + eps * x_adv.grad.sign()
    x_adv = torch.clamp(x_adv, 0, 1)

    return x_adv.detach()

def batch_fgsm_attack(model, batch_size, test_loader, eps=1e-4, save_imgs=True, save_dir='imgs'):
    os.makedirs(save_dir, exist_ok=True)

    right = 0
    success = 0
    total = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.eval()
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        if save_imgs:
            #save the original images
            for i in range(x_batch.size(0)):
                img = x_batch
                img = img[i].cpu().numpy().transpose(1, 2, 0)
                if img.shape[2] == 1:
                    img = img.squeeze(axis=2)
                img = np.clip(img, 0, 1)
                plt.imsave(f'{save_dir}/original_{batch_idx * batch_size + i}_label_{y_batch[i].item()}.png', img, cmap='gray')

        with torch.no_grad():
            outputs = model(x_batch)
            _, predicted = torch.max(outputs.data, 1)
            right += (predicted == y_batch).sum().item()
            right_mask = (predicted == y_batch)

        model.train()
        x_advs = fgsm_untargeted(model, x_batch, y_batch, eps)
        model.eval()

        with torch.no_grad():
            outputs = model(x_advs)
            _, predicted = torch.max(outputs.data, 1)
            # Calculate success rate
            wrong_mask = (predicted != y_batch)
            success += (right_mask & wrong_mask).sum().item()
            total += x_batch.size(0)
        
        if save_imgs:
            for i in range(x_batch.size(0)):
                img = x_advs[i].cpu().numpy().transpose(1, 2, 0)  # (H, W, C)
                if img.shape[2] == 1:
                    img = img.squeeze(axis=2)  # (H, W)
                img = np.clip(img, 0, 1)
                plt.imsave(
                    f'{save_dir}/adv_{batch_idx * batch_size + i}_label_{y_batch[i].item()}_untargeted__pred_{predicted[i].item()}.png',
                    img,
                    cmap='gray'
                )

    accuracy = (right / total) * 100
    logger.info(f'Accuracy: {accuracy:.2f}%')
    success_rate = (success / total) * 100
    logger.info(f'Untargeted FGSM attack success rate: {success_rate:.2f}%')
    return success_rate

# Data loader
batch_size = 8192
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

fgsm_eps = 3e-2
save_imgs = False

# MLP
model = MLP300K()
mnist_train_model(model, train_loader, test_loader, num_epochs=50, lr=0.003, save_path='model_mlp.ckpt')
# model.load_state_dict(torch.load('model_mlp.ckpt'))
batch_fgsm_attack(model, batch_size, test_loader, eps=fgsm_eps, save_imgs=save_imgs, save_dir='imgs/imgs_mlp')