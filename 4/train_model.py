from tqdm.auto import tqdm

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

import sys
sys.path.append("alpha-beta-CROWN/complete_verifier")  # Adjust path if necessary
from model_defs import ResNet18 as resnet18

model = resnet18()

transform = ToTensor()
train_data = CIFAR10(root="data", train=True, download=True)
train_data.transform = transform
train_loader = DataLoader(train_data, batch_size=2048, shuffle=True)

test_data = CIFAR10(root="data", train=False, download=True)
test_data.transform = transform
test_loader = DataLoader(test_data, batch_size=2048, shuffle=False)

criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=1e-2, momentum=0.9)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

torch.save(model.state_dict(), "resnet18_cifar.pth")

input_tensor = torch.rand((1, 3, 32, 32), dtype=torch.float32)
input_tensor = input_tensor.to("cuda")

torch.onnx.export(
    model,
    input_tensor,
    "resnet18_cifar.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=13
)