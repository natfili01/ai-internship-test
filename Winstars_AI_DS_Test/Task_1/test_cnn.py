from models.cnn_classifier import CNNMnist
import torch

# Create an instance of CNN
cnn_classifier = CNNMnist(input_channels=1, num_classes=10, learning_rate=0.001)

# Generate test data (10 grayscale images of size 28x28)
X_train = torch.rand(10, 1, 28, 28)  # Training images
y_train = torch.randint(0, 10, (10,))  # Random labels (digits 0-9)
y_train = torch.nn.functional.one_hot(y_train, num_classes=10).float()  # Convert to one-hot encoding

X_test = torch.rand(3, 1, 28, 28)  # Test images

# Train the model
cnn_classifier.train(X_train, y_train, epochs=5)

# Make predictions
predictions = cnn_classifier.predict(X_test)
print(predictions)  # Output predicted classes (digits 0-9)
