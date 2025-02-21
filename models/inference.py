import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras  # Ensure keras is imported

# Load the model in the correct format
model_path = r"C:\Users\ajuli\Documents\AI Test\Winstars_AI_DS_Test\Task_2\models\cnn_model.keras"
model = keras.models.load_model(model_path)

print(" Model successfully loaded from models/cnn_model.keras")

# Set test data directory
test_dir = r"C:\Users\ajuli\Documents\AI Test\Winstars_AI_DS_Test\Task_2\datasets_split\test"

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Prevent shuffling for accurate comparison
)

# Get true labels
y_true = test_generator.classes
class_indices = list(test_generator.class_indices.keys())  # Class names

# Perform predictions
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# Print classification report
print(" Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_indices))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_indices, yticklabels=class_indices)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# Function to classify a single image
def classify_animal(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    predicted_label = class_indices[predicted_index]
    confidence = predictions[0][predicted_index]  # Confidence score

    print(f"Prediction: {predicted_label} (Confidence: {confidence:.2f})")
    return predicted_label
