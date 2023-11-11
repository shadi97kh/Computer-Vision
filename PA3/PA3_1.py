import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define Fully Connected Autoencoder
class FCAutoencoder(nn.Module):
    def __init__(self):
        super(FCAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)

# Define Convolutional Autoencoder
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Initialize models and optimizers
fc_model = FCAutoencoder()
conv_model = ConvAutoencoder()

fc_optimizer = optim.Adam(fc_model.parameters(), lr=1e-3)
conv_optimizer = optim.Adam(conv_model.parameters(), lr=1e-3)

# Loss function
criterion = nn.MSELoss()

# Training function
def train_model(model, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, _ in train_loader:
            optimizer.zero_grad()
            reconstructed = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}')

# Train models
print("Training Fully Connected Autoencoder")
train_model(fc_model, fc_optimizer)

print("Training Convolutional Autoencoder")
train_model(conv_model, conv_optimizer)

# Function to Count the number of params
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print number of parameters for Fully Connected Autoencoder
fc_params = count_parameters(fc_model)
print(f"Number of parameters in Fully Connected Autoencoder: {fc_params}")

# Print number of parameters for Convolutional Autoencoder
conv_params = count_parameters(conv_model)
print(f"Number of parameters in Convolutional Autoencoder: {conv_params}")

# Function to display images
def display_images(model, data_loader, title):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(data_loader))
        output = model(data)
        output = output.view(-1, 1, 28, 28)
        output = output.numpy()

    fig, axes = plt.subplots(nrows=2, ncols=20, figsize=(10, 3))
    fig.suptitle(title)

    for ax, original, reconstructed in zip(axes.flatten(), data.numpy(), output):
        ax.imshow(original.squeeze(), cmap='gray' if original.shape[0] == 1 else None)
        ax.axis('off')

    for ax, reconstructed in zip(axes[1], output):
        ax.imshow(reconstructed.squeeze(), cmap='gray')
        ax.axis('off')

    plt.show()

# Displaying images for Fully Connected Autoencoder
display_images(fc_model, test_loader, "Fully Connected Autoencoder")

# Displaying images for Convolutional Autoencoder
display_images(conv_model, test_loader, "Convolutional Autoencoder")