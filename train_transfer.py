
import tensorflow as tf
from pathlib import Path

from data_loader import train_dataset, validation_dataset
from flower_mapping import FLOWER_CLASSES
from model_transfer import build_transfer_model

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_DIR = Path("dataset/split_v2/train")

# Calculate class weights from folder counts
class_counts = [
    len(list((TRAIN_DIR / flower).glob("*")))
    for flower in FLOWER_CLASSES
]

total_images = sum(class_counts)

class_weights = {
    index: total_images / (len(FLOWER_CLASSES) * count)
    for index, count in enumerate(class_counts)
    if count > 0
}

print("Class weights:", class_weights)

# Build transfer learning model
model = build_transfer_model()

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_DIR / "bloomscan_transfer_fixed_best.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.000001,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        "transfer_fixed_history.csv"
    )
]

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# Save the best model
best_model = tf.keras.models.load_model(
    MODEL_DIR / "bloomscan_transfer_fixed_best.keras"
)

best_model.save(
    MODEL_DIR / "bloomscan_transfer_fixed_final.keras"
)

print("\nSUCCESS: BloomScan Transfer Learning Training Completed!")

