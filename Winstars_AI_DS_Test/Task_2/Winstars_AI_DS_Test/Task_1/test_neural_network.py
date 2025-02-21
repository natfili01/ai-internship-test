from models.neural_network_classifier import FeedForwardNN
import torch

# Create an instance of the neural network
nn_classifier = FeedForwardNN(input_size=5, hidden_size=10, output_size=2, learning_rate=0.01)

# Generate test data (10 samples, each with 5 features)
X_train = torch.rand(10, 5)  # Training data
y_train = torch.randint(0, 2, (10,))  # Random labels (0 or 1)
y_train = torch.nn.functional.one_hot(y_train, num_classes=2).float()  # Convert to one-hot encoding

X_test = torch.rand(3, 5)  # Test data

# Train the model
nn_classifier.train(X_train, y_train, epochs=200)

# Make predictions
predictions = nn_classifier.predict(X_test)
print(predictions)  # Output predicted classes (0 or 1)

