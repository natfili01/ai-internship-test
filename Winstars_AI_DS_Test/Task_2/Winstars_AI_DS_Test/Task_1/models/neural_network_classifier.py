# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from models.mnist_classifier_interface import MnistClassifierInterface

class FeedForwardNN(nn.Module, MnistClassifierInterface):
    
    def __init__(self, input_size=5, hidden_size=10, output_size=2, learning_rate=0.01):
        super().__init__()

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

    def train(self, train_data, train_labels, epochs=100):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.forward(train_data)
            loss = self.criterion(outputs, train_labels)
            loss.backward()
            self.optimizer.step()

    def predict(self, test_data):
        with torch.no_grad():
            outputs = self.forward(test_data)
            return torch.argmax(outputs, dim=1)
