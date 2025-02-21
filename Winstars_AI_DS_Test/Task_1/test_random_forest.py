from models.random_forest_classifier import RandomForestMnist
import numpy as np

# Create the model object
rf_classifier = RandomForestMnist(n_estimators=10)

# Generate test data (10 samples with 5 features each)
X_train = np.random.rand(10, 5)   # Training data
y_train = np.random.randint(0, 2, 10)  # Random labels (0 or 1)

X_test = np.random.rand(3, 5)  # Test data

# Train the model
rf_classifier.train(X_train, y_train)

# Make predictions
predictions = rf_classifier.predict(X_test)
print(predictions)  # Output the predicted classes (0 or 1)
