
import tensorflow as tf
import numpy as np

from pathlib import Path
from data_loader import train_dataset, validation_dataset
from flower_mapping import FLOWER_CLASSES
from model import build_cnn_model

# Create model directory
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Build a completely fresh CNN model
model = build_cnn_model()

print("\nFresh BloomScan CNN model created successfully!")

# Calculate class weights
class_counts = np.zeros(len(FLOWER_CLASSES), dtype=np.int64)

for _, labels in train_dataset:
    class_counts += np.bincount(
        labels.numpy(),
        minlength=len(FLOWER_CLASSES)
    )

total_images = np.sum(class_counts)

class_weights = {
    index: total_images / (len(FLOWER_CLASSES) * count)
    for index, count in enumerate(class_counts)
    if count > 0
}

print("\n" + "=" * 50)
print("       BLOOMSCAN V2 TRAINING")
print("=" * 50)

print("\nTraining images per class:")
for index, flower in enumerate(FLOWER_CLASSES):
    print(f"{flower}: {class_counts[index]}")

print("\nClass weights calculated successfully.")

# Training callbacks
callbacks = [

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "bloomscan_v2_best.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_lr=0.000001,
        verbose=1
    )
]

# Start fresh training
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    initial_epoch=0,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks
)

# Load the best-performing v2 checkpoint
best_model = tf.keras.models.load_model(
    MODEL_DIR / "bloomscan_v2_best.keras"
)

# Save best model as final model
best_model.save(MODEL_DIR / "bloomscan_v2_final.keras")

print("\n" + "=" * 50)
print("       TRAINING COMPLETED")
print("=" * 50)

print("Best model: models/bloomscan_v2_best.keras")
print("Final model: models/bloomscan_v2_final.keras")
print("Training completed successfully!")
