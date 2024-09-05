import torch


class CNN(torch.nn.Module):
    def __init__(self, num_blocks:int, width_scale: float, num_classes:int):
        super(CNN, self).__init__()
        self.blocks = []
        in_channels = 1
        width = int(width_scale*4)
        out_channels = width
        for _ in range(num_blocks):
            self.blocks.append(Block(in_channels=in_channels, width=width, out_channels=out_channels))
            in_channels = out_channels
            width *= 2
            out_channels=width

        self.fc = torch.nn.Linear(in_features=int(width/2), out_features=num_classes)

    def forward(self, input: torch.Tensor)->torch.Tensor:
        x = input
        for block in self.blocks:
            x = block(x)
        x = torch.nn.functional.avg_pool2d(x, kernel_size=x.shape[-1])
        x = x.view(x.shape[0], x.shape[1])
        return self.fc(x) # returns logits

class Block(torch.nn.Module):
    def __init__(self, in_channels: int, width: int, out_channels: int):
        super(Block, self).__init__()
        self.activation = torch.nn.ReLU()
        self.cnn1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=width, kernel_size=3, padding="same", bias=False)
        self.cnn2 = torch.nn.Conv2d(in_channels=width, out_channels=out_channels, kernel_size=3, padding="same", bias=False)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=2)
    def forward(self, input: torch.Tensor)->torch.Tensor:
        x = self.cnn1(input)
        x = self.activation(x)
        x = self.cnn2(x)
        return self.max_pool(x)
        