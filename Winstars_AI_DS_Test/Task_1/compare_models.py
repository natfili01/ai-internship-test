import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from models.random_forest_classifier import RandomForestMnist
from models.neural_network_classifier import FeedForwardNN
from models.cnn_classifier import CNNMnist



# Normalize data: Convert pixel values from [0,255] to [0,1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and standard deviation of MNIST
])

# Load the MNIST dataset with normalization
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)


# Convert dataset to DataLoader for batch processing
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

# Extract the first batch for training and testing
X_train, y_train = next(iter(train_loader))
X_test, y_test = next(iter(test_loader))

# Convert labels to one-hot encoding for FeedForwardNN only
y_train_one_hot = torch.nn.functional.one_hot(y_train, num_classes=10).float()
y_test_one_hot = torch.nn.functional.one_hot(y_test, num_classes=10).float()

# Convert labels for CNN (must be integer class indices)
y_train_cnn = y_train.long()
y_test_cnn = y_test.long()


# Train and test Random Forest (Flattened images for compatibility)
X_train_flat = X_train.view(X_train.shape[0], -1).numpy()
X_test_flat = X_test.view(X_test.shape[0], -1).numpy()
y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

rf_model = RandomForestMnist(n_estimators=10)
rf_model.train(X_train_flat, y_train_np)
rf_predictions = rf_model.predict(X_test_flat)
rf_accuracy = accuracy_score(y_test_np, rf_predictions)
print(f"Random Forest Accuracy on MNIST: {rf_accuracy:.2f}")

# Train and test FeedForward Neural Network
ffnn_model = FeedForwardNN(input_size=28*28, hidden_size=128, output_size=10)
ffnn_model.train(X_train.view(X_train.shape[0], -1), y_train_one_hot, epochs=20)
ffnn_predictions = ffnn_model.predict(X_test.view(X_test.shape[0], -1))
ffnn_accuracy = accuracy_score(y_test_np, ffnn_predictions.numpy())
print(f"FeedForward Neural Network Accuracy on MNIST: {ffnn_accuracy:.2f}")

# Train and test CNN Model
cnn_model = CNNMnist(input_channels=1, num_classes=10)
cnn_model.train(X_train, y_train_cnn, epochs=20)  #Use integer labels for CNN
cnn_predictions = cnn_model.predict(X_test)
cnn_accuracy = accuracy_score(y_test_cnn.numpy(), cnn_predictions.numpy())
print(f"Convolutional Neural Network Accuracy on MNIST: {cnn_accuracy:.2f}")
