import torch
import torch.nn as nn
import torch.optim as optim
from models.mnist_classifier_interface import MnistClassifierInterface

class CNNMnist(nn.Module, MnistClassifierInterface):
    
    def __init__(self, input_channels=1, num_classes=10, learning_rate=0.001):
        super().__init__()

        # Convolutional layers to extract features from images
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Compute the flattened size dynamically for the fully connected layer
        self.flatten_size = 128 * 3 * 3  
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)  # Reduce learning rate every 5 epochs

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.shape[0], -1)  # Flatten before fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  
        return x  

    def train(self, train_data, train_labels, epochs=5):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.forward(train_data)
            loss = self.criterion(outputs, train_labels)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()  

    def predict(self, test_data):
        with torch.no_grad():  
            outputs = self.forward(test_data)  
            return torch.argmax(outputs, dim=1)  # Get the predicted class
