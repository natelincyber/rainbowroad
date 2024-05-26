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

        self.PALM_POSITION = nn.Linear(64, 1)
        self.TH_KNU3_A = nn.Linear(64, 1)
        self.TH_KNU2_A = nn.Linear(64, 1)
        self.TH_KNU1_B = nn.Linear(64, 1)
        self.TH_KNU1_A = nn.Linear(64, 1)

        self.F2_KNU3_A = nn.Linear(64, 1)
        self.F2_KNU2_A = nn.Linear(64, 1)
        self.F2_KNU1_B = nn.Linear(64, 1)
        self.F2_KNU1_A = nn.Linear(64, 1)

        self.F1_KNU3_A = nn.Linear(64, 1)
        self.F1_KNU2_A = nn.Linear(64, 1)
        self.F1_KNU1_B = nn.Linear(64, 1)
        self.F1_KNU1_A = nn.Linear(64, 1)

        self.F3_KNU3_A = nn.Linear(64, 1)
        self.F3_KNU2_A = nn.Linear(64, 1)
        self.F3_KNU1_B = nn.Linear(64, 1)
        self.F3_KNU1_A = nn.Linear(64, 1)

        self.F4_KNU3_A = nn.Linear(64, 1)
        self.F4_KNU2_A = nn.Linear(64, 1)
        self.F4_KNU1_B = nn.Linear(64, 1)
        self.F4_KNU1_A = nn.Linear(64, 1)


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

        # Output layers
        palm = self.PALM_POSITION(x)
        th_knu3_a = self.TH_KNU3_A(x)
        th_knu2_a = self.TH_KNU2_A(x)
        th_knu1_b = self.TH_KNU1_B(x)
        th_knu1_a = self.TH_KNU1_A(x)

        f2_knu3_a = self.F2_KNU3_A(x)
        f2_knu2_a = self.F2_KNU2_A(x)
        f2_knu1_b = self.F2_KNU1_B(x)
        f2_knu1_a = self.F2_KNU1_A(x)

        f1_knu3_a = self.F1_KNU3_A(x)
        f1_knu2_a = self.F1_KNU2_A(x)
        f1_knu1_b = self.F1_KNU1_B(x)
        f1_knu1_a = self.F1_KNU1_A(x)

        f3_knu3_a = self.F3_KNU3_A(x)
        f3_knu2_a = self.F3_KNU2_A(x)
        f3_knu1_b = self.F3_KNU1_B(x)
        f3_knu1_a = self.F3_KNU1_A(x)

        f4_knu3_a = self.F4_KNU3_A(x)
        f4_knu2_a = self.F4_KNU2_A(x)
        f4_knu1_b = self.F4_KNU1_B(x)
        f4_knu1_a = self.F4_KNU1_A(x)

        return palm, th_knu3_a, th_knu2_a, th_knu1_a, th_knu1_b, \
        f2_knu3_a, f2_knu2_a, f2_knu1_a, f2_knu1_b, f1_knu3_a, f1_knu2_a, \
        f1_knu1_a, f1_knu1_b, f3_knu3_a, f3_knu2_a, f3_knu1_a, f3_knu1_b, \
        f4_knu3_a, f4_knu2_a, f4_knu1_a, f4_knu1_b