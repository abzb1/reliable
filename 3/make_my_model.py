import random

from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import numpy as np
import torch

class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 10),
            torch.nn.Sigmoid(),
            torch.nn.Linear(10, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def generate_data(num_samples):
    data = []
    for _ in range(num_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, 40)
        z = random.uniform(20, 40)
        if z == 0:
            z = 1e-6
        target = (x*2 + y / z)
        data.append((x, y, z, target))
    return data    

class DS(torch.utils.data.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.data = generate_data(self.num_samples)

        self.data = [[torch.tensor([x, y, z], dtype=torch.float32).to("cuda"), torch.tensor(target, dtype=torch.float32).to("cuda")] for x, y, z, target in tqdm(self.data)]

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        if idx >= len(self.data):
            raise IndexError("Index out of range")
        return self.data[idx]
    
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

my_net = MyNet()
my_net = my_net.to("cuda")

optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

dataset = DS(num_samples = 1_000_000)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8192, shuffle=True)

losses = []
for epoch in tqdm(range(100)):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = my_net(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss = torch.sqrt(loss)
        loss.backward()
        optimizer.step()
    
    losses.append(loss.item())

plt.yscale('log')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(losses)
plt.savefig('loss.png')

print(f"Final loss: {losses[-1]:.3e}")

input_tensor = torch.rand((1, 3), dtype=torch.float32)
input_tensor = input_tensor.to("cuda")

torch.onnx.export(
    my_net,
    (input_tensor,),
    "my_model.onnx",
    input_names=["input"],
    dynamo=False
)