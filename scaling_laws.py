import torch
import torchvision
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from cnn import CNN


if(__name__=="__main__"):
    train_data = torchvision.datasets.MNIST(root="./", train=True, transform=torchvision.transforms.ToTensor(), download=True)
    train_data = torchvision.datasets.MNIST(root="./", train=False, transform=torchvision.transforms.ToTensor(), download=True)

    train_loader = DataLoader(train_data, shuffle=True, batch_size=16)
   

    num_widths=2
    for i in range(num_widths):
        cnn = CNN(num_blocks=2, width_scale=math.sqrt(2)**i, num_classes=10)


    # train
    optim = torch.optim.Adam(params=cnn.parameters())
    num_epochs = 10
    for epoch in range(num_epochs):
        avg_loss = 0.
        for i, (img,target) in enumerate(train_loader):
            optim.zero_grad()
            output = cnn(img)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
            optim.step()
            avg_loss += loss
        avg_loss /= i
        print("Avg. loss: ", avg_loss)
