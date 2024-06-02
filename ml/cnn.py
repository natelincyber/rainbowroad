import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=1)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=1)
        self.relu5 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=5, stride=5, padding=1)
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1681, 64)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(64, 64)
        self.relu7 = nn.ReLU()

        self.output = nn.Linear(64, 42) # 21 outputs * 2 coordinates


    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x) 
        x = self.relu3(x)
        x = self.pool(x) # no memory D: 
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        # Fully connected layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)

        outputs = self.output(x)
        outputs = outputs.view(-1, 21, 2)

        return outputs

if __name__ == '__main__':
    net = Net()
    x = torch.randn(1, 3, 224, 224) # test a simple 224x224 3 channel image in single batch
    y = net(x)
    print(y[0])
    print(y.shape)