import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info and warnings

# Load train and validation data
train_data = pd.read_csv('C:\\Skinalyze-main\\train.txt', delimiter='\t', header=None, names=['image_path', 'label'])
valid_data = pd.read_csv('C:\\Skinalyze-main\\valid.txt', delimiter='\t', header=None, names=['image_path', 'label'])

# Define image size and batch size
img_size = (128, 128)
batch_size = 32

# Map labels to numeric values
label_to_num = {'dry': 0, 'normal': 1, 'oily': 2}

# Add numeric labels to the datasets
train_data['numeric_label'] = train_data['label'].map(label_to_num)
valid_data['numeric_label'] = valid_data['label'].map(label_to_num)

# Data generators with augmentation for training and rescaling for validation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalize pixel values
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1.0/255)

# Create the training generator
train_generator = train_datagen.flow_from_dataframe(
    train_data,
    x_col='image_path',
    y_col='label',  # Use categorical labels
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  # Multi-class classification
)

# Create the validation generator
valid_generator = valid_datagen.flow_from_dataframe(
    valid_data,
    x_col='image_path',
    y_col='label',  # Use categorical labels
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Build the CNN model
model = Sequential([
    Input(shape=(128, 128, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),  # Deeper layer
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),  # Increased dense layer units
    Dropout(0.5),  # Dropout to prevent overfitting
    Dense(3, activation='softmax')  # Output layer for 3 classes
])

# Compile the model
optimizer = Adam(learning_rate=0.0001)  # Reduced learning rate for better convergence
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_data) // batch_size,
    validation_data=valid_generator,
    validation_steps=len(valid_data) // batch_size,
    epochs=20,  # Increased epochs for better learning
    verbose=1  # Use verbose=1 for progress output
)

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(valid_generator, steps=len(valid_data) // batch_size, verbose=1)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

# Save the trained model
model.save('skin_type_model.keras')

# Optional: Plot training history for accuracy and loss
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
