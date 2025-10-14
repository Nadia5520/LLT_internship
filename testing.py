# train_model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import os

# 1. SET YOUR DATASET PATH HERE (THIS IS THE ONLY LINE YOU MUST CHANGE)
data_dir = "G:\My Drive\Library_Internship_Artworks_Project\Margo_Veillon_Dataset"  

# 2. Setup parameters
batch_size = 30
img_height = 224
img_width = 224

# 3. Load and prepare the data
print("Loading images...")
train_datagen = ImageDataGenerator(
    rescale=1./255,        # Normalize pixel values
    validation_split=0.2,  # Hold back 20% of data for validation
    horizontal_flip=True   # Simple data augmentation
)

train_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

val_ds = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

# 4. Build the model using Transfer Learning
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze the pre-trained base

model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid') # Single output node for binary classification
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 5. Train the model
print("\nStarting training...")
history = model.fit(
    train_ds,
    epochs=10,
    validation_data=val_ds
)

# 6. Plot the results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

# 7. Save the trained model for later use
model.save('margo_veillon_classifier.keras')
print("Model saved as 'margo_veillon_classifier.keras'")