from tqdm.auto import tqdm
from loguru import logger

import torch
from torch import nn
from torch import optim

def cifar10_train_model(model, train_loader, test_loader, num_epochs=50, lr=0.003, save_path='model.ckpt'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    for _ in tqdm(range(num_epochs)):
        model.train()
        for images, labels in tqdm(train_loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the model on the 10000 test images: {accuracy:.2f} %')

    torch.save(model.state_dict(), save_path)
    return accuracy

def cifar10_test_model(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        logger.info(f'Accuracy of the model on the 10000 test images: {accuracy:.2f} %')

    return accuracy