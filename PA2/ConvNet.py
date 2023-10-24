import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, selected_mode):
        super(ConvNet, self).__init__()

        # Layers for Model 1
        self.fc_layer1 = nn.Linear(28 * 28, 100)
        self.activation_sigmoid = nn.Sigmoid()

        # Layers for Models 2 & 3: Convolutional Layers
        self.conv_layer1 = nn.Conv2d(1, 40, kernel_size=5, stride=1)
        self.conv_layer2 = nn.Conv2d(40, 40, kernel_size=5, stride=1)
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer for Model 2, 3, and 4
        self.fc_layer2 = nn.Linear(40 * 4 * 4, 100)
        self.fc_layer3 = nn.Linear(100, 100)  # Additional for Model 4

        # Layers for Model 5
        self.fc_layer4 = nn.Linear(40 * 4 * 4, 1000)
        self.fc_layer5 = nn.Linear(1000, 1000)

        # Dropout for regularization
        self.regularization_dropout = nn.Dropout(0.5)

        # Activation functions
        self.activation_relu = nn.ReLU()

        # Output layer common to all models
        self.output_layer = nn.Linear(100, 10)
        self.output_layer_expanded = nn.Linear(1000, 10)  # For Model 5

        # Check if the provided mode is valid
        if selected_mode not in [1, 2, 3, 4, 5]:
            print("Invalid mode", selected_mode, "selected. Select between 1-5")
            exit(0)
        else:
            self.mode = selected_mode

    def forward(self, input_tensor):
        # Forward pass based on the selected model mode
        if self.mode == 1:
            return self.model_1(input_tensor)
        elif self.mode == 2:
            return self.model_2(input_tensor)
        elif self.mode == 3:
            return self.model_3(input_tensor)
        elif self.mode == 4:
            return self.model_4(input_tensor)
        else:
            return self.model_5(input_tensor)

    # Define each model as mentioned
    def model_1(self, input_tensor):
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = self.fc_layer1(input_tensor)
        input_tensor = self.activation_sigmoid(input_tensor)
        output = self.output_layer(input_tensor)
        return output

    def model_2(self, input_tensor):
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = self.fc_layer1(input_tensor)
        input_tensor = self.activation_sigmoid(input_tensor)
        output = self.output_layer(input_tensor)
        return output

    def model_3(self, input_tensor):
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = self.fc_layer1(input_tensor)
        input_tensor = self.activation_sigmoid(input_tensor)
        output = self.output_layer(input_tensor)
        return output

    def model_4(self, input_tensor):
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = self.fc_layer1(input_tensor)
        input_tensor = self.activation_sigmoid(input_tensor)
        output = self.output_layer(input_tensor)
        return output

    def model_5(self, input_tensor):
        input_tensor = input_tensor.view(input_tensor.size(0), -1)
        input_tensor = self.fc_layer1(input_tensor)
        input_tensor = self.activation_sigmoid(input_tensor)
        output = self.output_layer(input_tensor)
        return output
