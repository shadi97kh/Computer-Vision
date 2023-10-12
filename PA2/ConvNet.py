import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, mode):
        super(ConvNet, self).__init__()

        # For STEP 1
        self.fc1 = nn.Linear(28 * 28, 100)  # Assuming input image is 28x28
        self.sigmoid = nn.Sigmoid()

        # For STEP 2 & 3
        self.conv1 = nn.Conv2d(1, 40, kernel_size=5, stride=1)  # Assuming grayscale image hence 1 input channel
        self.conv2 = nn.Conv2d(40, 40, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # For STEP 4 & 5
        self.fc2 = nn.Linear(100, 100)
        self.fc3_large = nn.Linear(100, 1000)  # For step 5
        self.fc4_large = nn.Linear(1000, 1000)  # For step 5

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        if mode not in [1, 2, 3, 4, 5]:
            print("Invalid mode ", mode, "selected. Select between 1-5")
            exit(0)
        else:
            self.mode = mode
        self.fc_out = nn.Linear(100, 10) # Output layer

    def forward(self, x):
        if self.mode == 1:
            return self.model_1(x)
        elif self.mode == 2:
            return self.model_2(x)
        elif self.mode == 3:
            return self.model_3(x)
        elif self.mode == 4:
            return self.model_4(x)
        else:
            return self.model_5(x)

    def model_1(self, X):
        X = X.view(X.size(0), -1)
        X = self.fc1(X)
        X = self.sigmoid(X)
        X = self.fc_out(X)
        return X

    def model_2(self, X):
        X = self.conv1(X)
        X = self.sigmoid(X)
        X = self.pool(X)
        X = self.conv2(X)
        X = self.sigmoid(X)
        X = self.pool(X)
        X = X.view(X.size(0), -1)
        X = self.fc1(X)
        X = self.sigmoid(X)
        X = self.fc_out(X)
        return X

    def model_3(self, X):
        X = self.conv1(X)
        X = self.relu(X)
        X = self.pool(X)
        X = self.conv2(X)
        X = self.relu(X)
        X = self.pool(X)
        X = X.view(X.size(0), -1)
        X = self.fc1(X)
        X = self.relu(X)
        X = self.fc_out(X)
        return X

    def model_4(self, X):
        X = self.model_3(X)
        X = self.fc2(X)
        X = self.relu(X)
        X = self.fc_out(X)
        return X

    def model_5(self, X):
        X = self.model_3(X)
        X = self.fc3_large(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.fc4_large(X)
        X = self.relu(X)
        X = self.dropout(X)
        X = self.fc_out(X)
        return X
