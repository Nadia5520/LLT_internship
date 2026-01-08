"""
train_classifier.py

Transfer-learning training pipeline for artwork classification (ResNet50).

Usage:
    python train_classifier.py --data_dir /path/to/data --output model.keras

Data layout (if using flow_from_directory or image_dataset_from_directory):
    /data_dir/
        class_a/
        class_b/
        ...

This script:
 - loads images with train/validation split
 - trains head (frozen base) then optionally fine-tunes top layers
 - saves final model and a feature-extractor model for embeddings
"""
import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import math

def build_model(input_shape=(224,224,3), num_classes=1, base_trainable=False):
    base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = base_trainable
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    # Optional lightweight augmentation layer (training only)
    data_aug = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),
        layers.RandomZoom(0.08),
    ])
    x = data_aug(x)
    x = tf.keras.applications.resnet.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model, base

def prepare_datasets(data_dir, img_size=(224,224), batch_size=32, val_split=0.2, seed=123):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size
    )
    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    return train_ds, val_ds, class_names

def compute_class_weights(train_ds, class_names):
    # For image_dataset_from_directory, we can count images for class weights
    counts = {c: 0 for c in class_names}
    for batch, labels in train_ds.unbatch().map(lambda x,y: (x,y)):
        # this loop is expensive for huge datasets. Instead, consider counting files on disk.
        pass
    # Fallback: count files on disk
    return None

def main(args):
    img_size = (args.img_height, args.img_width)
    train_ds, val_ds, class_names = prepare_datasets(args.data_dir, img_size=img_size, batch_size=args.batch_size, val_split=args.val_split)
    num_classes = 1 if len(class_names) == 2 else len(class_names)
    print("Classes:", class_names)
    model, base = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes, base_trainable=False)

    # Compile head
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss='binary_crossentropy' if num_classes==1 else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, "best_model.keras")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
    ]

    initial_epochs = args.initial_epochs
    print(f"Training head for {initial_epochs} epochs...")
    history = model.fit(train_ds, epochs=initial_epochs, validation_data=val_ds, callbacks=callbacks)

    # Fine-tune: unfreeze last N layers of base
    if args.fine_tune:
        base.trainable = True
        # Freeze all layers except the last `fine_tune_at` layers
        if args.fine_tune_at > 0:
            for layer in base.layers[:-args.fine_tune_at]:
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
                      loss='binary_crossentropy' if num_classes==1 else 'sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        fine_epochs = args.fine_tune_epochs
        total_epochs = initial_epochs + fine_epochs
        print(f"Fine-tuning for {fine_epochs} epochs (total {total_epochs})...")
        history_f = model.fit(train_ds, epochs=total_epochs, initial_epoch=history.epoch[-1]+1,
                              validation_data=val_ds, callbacks=callbacks)

    # Save final model
    final_path = os.path.join(args.output_dir, args.output_name)
    print("Saving model to", final_path)
    model.save(final_path)

    # Save feature-extractor model for embeddings (strip final activation)
    feat_model = tf.keras.Model(model.input, model.layers[-3].output)  # layer before final dense/dropout may vary
    feat_path = os.path.join(args.output_dir, "feature_extractor.keras")
    try:
        feat_model.save(feat_path)
        print("Saved feature extractor to", feat_path)
    except Exception as e:
        print("Warning: could not save feature extractor:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="models", help="Where to save checkpoints & model")
    parser.add_argument("--output_name", type=str, default="margo_classifier.keras", help="Final model filename")
    parser.add_argument("--img_height", type=int, default=224)
    parser.add_argument("--img_width", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--initial_epochs", type=int, default=8)
    parser.add_argument("--fine_tune", action="store_true", help="Whether to run fine-tuning stage")
    parser.add_argument("--fine_tune_at", type=int, default=50, help="Unfreeze last N layers of base model")
    parser.add_argument("--fine_tune_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)
    args = parser.parse_args()
    main(args)