import random
import itertools

import numpy as np
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10
from torchvision.models import resnet50

from utils import cifar10_train_model, cifar10_test_model

# MNIST dataset
transform_pipeline = ToTensor()
train_dataset = CIFAR10(root='data/', train=True, transform=transform_pipeline, download=True)
test_dataset = CIFAR10(root='data/', train=False, transform=transform_pipeline)

# Data loader
batch_size = 8192
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

fgsm_eps = 1e-3
save_imgs = False

# train 1

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model1 = resnet50(num_classes=10)
# cifar10_train_model(model1, train_loader, test_loader, num_epochs=50, lr=0.003, save_path='model_1.ckpt')
model1.load_state_dict(torch.load('model_1.ckpt'))
cifar10_test_model(model1, test_loader)

# train 2

seed = 1784
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model2 = resnet50(num_classes=10)
# cifar10_train_model(model2, train_loader, test_loader, num_epochs=50, lr=0.003, save_path='model_2.ckpt')
model2.load_state_dict(torch.load('model_2.ckpt'))
cifar10_test_model(model2, test_loader)

seed = 3123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

seed_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
models = [model1, model2]
for m in models:
    m.eval()

class Coverage:
    def __init__(self, model, threshold=0.0):
        self.threshold = threshold
        self.total = 0
        self.hit   = set()
        self._register(model)

    def _register(self, model):
        for layer in model.modules():
            if isinstance(layer, (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh)):
                layer.register_forward_hook(self._hook)

    def _hook(self, layer, inp, out):
        bs = out.shape[0]
        flat = out.detach().view(bs, -1)
        active = (flat > self.threshold)
        for b in range(bs):
            idx = torch.nonzero(active[b]).squeeze(1).add(self.total)
            self.hit.update(idx.tolist())
        self.total += flat.shape[1]

    def ratio(self):
        return len(self.hit) / max(1, self.total)

cover_trackers = [Coverage(m) for m in models]

def joint_objective(x, ref_idx, target_cls, lam1, lam2, neuron_idx=None):
    outs = [m(x) for m in models]
    obj1 = sum(o[0,target_cls] for i,o in enumerate(outs) if i!=ref_idx) - lam1*outs[ref_idx][0,target_cls]
    obj2 = 0
    if neuron_idx is not None:
        mdl_i, n_idx = neuron_idx
        penultimate = models[mdl_i].penultimate(x)
        obj2 = penultimate.view(-1)[n_idx]
    return obj1 + lam2*obj2

@torch.no_grad()
def pick_uncovered_neuron():
    choices = [(i, list(set(range(cv.total)) - cv.hit))
               for i,cv in enumerate(cover_trackers) if cv.total>0]
    choices = [(i, n) for i,neus in choices for n in neus]
    return random.choice(choices) if choices else None

epsilon = 1e-3
alpha   = 1e-2

def deepxplore(seed_loader, lam1=1.0, lam2=0.5,
                    alpha=alpha, epsilon=epsilon,
                    max_iters=100, coverage_goal=1.0):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generated = []

    for (x0, _) in tqdm(itertools.cycle(seed_loader), dynamic_ncols=True):
        x0     = x0.to(device)
        x_adv  = x0.clone().detach().requires_grad_(True)
        preds  = [torch.argmax(m(x_adv)) for m in models]
        target = preds[0].item()
        if len(set(preds)) != 1:
            continue

        ref_idx = random.randrange(len(models))

        for itr in range(max_iters):
            loss = -joint_objective(x_adv, ref_idx, target, lam1, lam2,
                                    neuron_idx=pick_uncovered_neuron())
            loss.backward()
            grad_sign = x_adv.grad.sign()

            with torch.no_grad():
                delta = torch.clamp(x_adv + alpha*grad_sign - x0,
                                    min=-epsilon, max=epsilon)
                x_adv = torch.clamp(x0 + delta, 0, 1).detach().requires_grad_(True)

            preds = [torch.argmax(m(x_adv)) for m in models]
            if len(set(preds)) > 1:
                generated.append(x_adv.cpu())
                for m in models:
                    m(x_adv)
                if all(cv.ratio() >= coverage_goal for cv in cover_trackers):
                    return generated
                break

    return generated

tests = deepxplore(seed_loader,
                        lam1=3, lam2=0.5,
                        alpha=1e-4, epsilon=1e-3,
                        max_iters=200, coverage_goal=1e-3)

print(f"found {len(tests)} diff-inducing inputs")