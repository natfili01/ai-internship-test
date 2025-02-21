import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Define Paths
data_dir = "C:/Users/ajuli/Documents/AI Test/Winstars_AI_DS_Test/Task_2/datasets_split/train"
val_dir = "C:/Users/ajuli/Documents/AI Test/Winstars_AI_DS_Test/Task_2/datasets_split/val"
model_save_path = "C:/Users/ajuli/Documents/AI Test/Winstars_AI_DS_Test/Task_2/models/cnn_model.h5"

# Data Augmentation (Reduced Aggressiveness)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
train_generator = train_datagen.flow_from_directory(
    data_dir,
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

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10-class classification
])

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.0005),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=30,  # Train for more epochs
    verbose=1
)

# Save Model
model.save(model_save_path)
print(f" Model saved at {model_save_path}")
