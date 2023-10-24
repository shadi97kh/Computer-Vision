import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        # Shared layers
        self.conv1 = nn.Conv2d(1, 40, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(40, 40, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # For STEP 1 and subsequent steps
        self.fc1 = nn.Linear(4 * 4 * 40, 100)
        
        # For STEP 4 and 5
        self.fc2 = nn.Linear(100, 100)
        self.fc3_large = nn.Linear(100, 1000)
        self.fc4_large = nn.Linear(1000, 1000)
        self.fc_out = nn.Linear(100, 10) 
        self.fc_out_large = nn.Linear(1000, 10)

        # Activation and regularization
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.mode = mode

    def forward(self, x):
        
        if self.mode == 1:
            x = self.pool(self.sigmoid(self.conv1(x)))
            x = self.pool(self.sigmoid(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.sigmoid(self.fc1(x))
            x = self.fc_out(x)
        elif self.mode == 2 or self.mode == 3:
            x = self.pool(self.sigmoid(self.conv1(x)) if self.mode == 2 else self.pool(self.relu(self.conv1(x))))
            x = self.pool(self.sigmoid(self.conv2(x)) if self.mode == 2 else self.pool(self.relu(self.conv2(x))))
            x = x.view(x.size(0), -1)
            x = self.sigmoid(self.fc1(x)) if self.mode == 2 else self.relu(self.fc1(x))
            x = self.fc_out(x)
        elif self.mode == 4:
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.fc_out(x)
        elif self.mode == 5:
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc3_large(x))
            x = self.dropout(x)
            x = self.relu(self.fc4_large(x))
            x = self.dropout(x)
            x = self.fc_out_large(x)
        return 