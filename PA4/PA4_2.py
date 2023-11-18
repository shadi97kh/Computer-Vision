import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class NetModified(nn.Module):
    def __init__(self):
        super(NetModified, self).__init__()
        # Increased Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # Additional layer
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)  # Additional layer
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Adjusted Fully connected layers
        # NOTE: The input features of fc1 might need adjustment based on the output size of conv5
        self.fc1 = nn.Linear(512, 1024)  # Adjusted for new conv layer output
        self.fc2 = nn.Linear(1024, 512)          # Adjusted for new input size
        self.fc3 = nn.Linear(512, 10)
        
        # Batch normalization
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)  # For the new layer
        self.batch_norm5 = nn.BatchNorm2d(512)  # For the new layer
        
        # Dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Sequence of convolutional and pool layers
        x = self.pool(self.batch_norm1(F.relu(self.conv1(x))))
        x = self.pool(self.batch_norm2(F.relu(self.conv2(x))))
        x = self.pool(self.batch_norm3(F.relu(self.conv3(x))))
        x = self.pool(self.batch_norm4(F.relu(self.conv4(x))))
        x = self.pool(self.batch_norm5(F.relu(self.conv5(x))))
        
        # Flatten the image
        x = x.view(-1, 512 * 1 * 1)
        
        # Fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

# Function to train and evaluate the model
import matplotlib.pyplot as plt

def train_and_evaluate(model, trainloader, testloader, criterion, optimizer, epochs=30):
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        model.train()  # Set the model to training mode
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = running_loss / len(trainloader)
        train_accuracy_percent = 100 * correct_train / total_train
        train_loss.append(avg_train_loss)
        train_accuracy.append(train_accuracy_percent)

        running_loss = 0.0
        correct_test = 0
        total_test = 0

        # Testing loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        avg_test_loss = running_loss / len(testloader)
        test_accuracy_percent = 100 * correct_test / total_test
        test_loss.append(avg_test_loss)
        test_accuracy.append(test_accuracy_percent)

        # Print epoch results
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy_percent:.2f}%')
        print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy_percent:.2f}%\n')

    # Plot training and testing losses
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(test_loss, label='Testing Loss')
    plt.title('Training and Testing Loss Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Print final accuracy scores
    final_train_accuracy = train_accuracy[-1]
    final_test_accuracy = test_accuracy[-1]
    print(f'Final Training Accuracy: {final_train_accuracy:.2f}%')
    print(f'Final Test Accuracy: {final_test_accuracy:.2f}%')

    return train_loss, test_loss, train_accuracy, test_accuracy

if __name__ == '__main__':
    # Define the CNN
    net = NetModified()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Transformations for the input data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Downloading and loading the CIFAR10 training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    # Train and evaluate the network
    train_and_evaluate(net, trainloader, testloader, criterion, optimizer)