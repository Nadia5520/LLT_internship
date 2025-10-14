# testing.py - DEBUGGING VERSION
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import os
import sys

# 1. SET YOUR DATASET PATH HERE
data_dir = "G:\My Drive\Library_Internship_Artworks_Project\Margo_Veillon_Dataset"  # <--- CHANGE THIS TO YOUR PATH
print(f"Data directory: {data_dir}")
print(f"This path exists: {os.path.exists(data_dir)}")

# 2. DEEP FOLDER ANALYSIS - Let's see EXACTLY what's in there
print("\n=== DEEP FOLDER ANALYSIS ===")
if os.path.exists(data_dir):
    for root, dirs, files in os.walk(data_dir):
        level = root.replace(data_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}[{os.path.basename(root)}/]")
        sub_indent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Print first 5 files in each folder
            print(f"{sub_indent}{file}")
        if len(files) > 5:
            print(f"{sub_indent}... and {len(files) - 5} more files")
else:
    print("ERROR: Path does not exist!")
    sys.exit()

# 3. Try to load data the same way Keras does
print("\n=== ATTEMPTING TO LOAD DATA ===")
try:
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    train_ds = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_ds = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    
    print("SUCCESS: Data loaded properly!")
    print(f"Training class indices: {train_ds.class_indices}")
    
    # 4. If successful, build and train model
    print("\n=== BUILDING MODEL ===")
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    model = Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        epochs=3,  # Just do 3 epochs for testing
        validation_data=val_ds
    )
    
    # 5. Save if successful
    model.save('margo_veillon_classifier.keras')
    print("Model saved successfully!")

except Exception as e:
    print(f"ERROR: {e}")
    print("\nThis error suggests the data loader couldn't find the expected folder structure.")
    print("Please double-check that your 'dataset' folder contains ONLY two subfolders:")
    print("1. 'margo_veillon/' - containing only her images")
    print("2. 'other/' - containing only other artists' images")
    print("There should be NO other files or folders in the main 'dataset' directory.")