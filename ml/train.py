import os
import sys
import torch
import torch.nn as nn
from cnn import Net
import torch.optim as optim
from dataloader import HandPose
from torchvision.transforms import v2
from torch.utils.data import DataLoader, random_split

learning_rate = 0.001
batch_size = 64
epochs = 1

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HandPose(filename="C:\\Users\\natel\\Dev\\Projects\\RainbowRoad\\data\\dataset.h5")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    # multithreaded data loading
    trainloader = DataLoader(train, batch_size=batch_size, 
                             shuffle=True)
    # valloader = DataLoader(val, batch_size=batch_size, 
    #                    shuffle=False, num_workers=2)
    
    net = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    print(f"Training on {device}")

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs,labels) in enumerate(trainloader,0):
            inputs = inputs.to(device)
            labels = labels.to(device)

            print(inputs)

            # #forward pass
            # outputs = net(inputs)

            # print(outputs)
        #     loss = criterion(outputs, labels)

        #     #backwards + optimize
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     running_loss = loss.item()
        #     if i % 100 == 99:
        #         print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
        #         running_loss = 0.0
                
        # torch.save(net.state_dict(), os.path.join("models", f"{path}_e{epoch}.pth"))


    print("Finished training") 
    # torch.save(net.state_dict(), os.path.join("models", f"{path}_final.pth"))

if __name__ == '__main__':
    train()
            