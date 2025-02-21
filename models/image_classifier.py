import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import os
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

#  Dataset Paths
train_dir = r"C:\Users\ajuli\Documents\AI Test\Winstars_AI_DS_Test\Task_2\datasets_split\train"
val_dir = r"C:\Users\ajuli\Documents\AI Test\Winstars_AI_DS_Test\Task_2\datasets_split\val"
test_dir = r"C:\Users\ajuli\Documents\AI Test\Winstars_AI_DS_Test\Task_2\datasets_split\test"

#  Get Class Weights (Handles Imbalanced Data)
classes = os.listdir(train_dir)
class_counts = {cls: len(os.listdir(os.path.join(train_dir, cls))) for cls in classes}
total_samples = sum(class_counts.values())

class_weight_dict = {
    i: total_samples / (len(classes) * class_counts[cls]) for i, cls in enumerate(classes)
}

print("Class Weights:", class_weight_dict)

# Updated Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,  # More rotation
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,  # More shear
    zoom_range=0.3,  # More zoom
    brightness_range=[0.7, 1.3],  # Random brightness
    horizontal_flip=True,
    vertical_flip=False,  # DO NOT flip vertically
    fill_mode="nearest"
)
# No Augmentation for Validation
val_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

# CNN Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  
    Dense(len(classes), activation='softmax')  
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,
    class_weight=class_weight_dict  #  Uses class balancing
)

# Save Model
model_path = r"C:\Users\ajuli\Documents\AI Test\Winstars_AI_DS_Test\Task_2\models\image_classifier_v3.h5"
model.save(model_path)
print(f" Model saved at {model_path}")
