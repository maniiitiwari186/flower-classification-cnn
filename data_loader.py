
import tensorflow as tf
from pathlib import Path

from flower_mapping import FLOWER_CLASSES

# Dataset configuration
IMAGE_SIZE = (160 , 160)
BATCH_SIZE = 2
SEED = 42

# Dataset directories
DATASET_DIR = Path("dataset/split_v2")

TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"

# Load training dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=FLOWER_CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

# Load validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    labels="inferred",
    label_mode="int",
    class_names=FLOWER_CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Load testing dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=FLOWER_CLASSES,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Normalize pixel values from 0-255 to 0-1
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization_layer(images), labels
    )
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (
        normalization_layer(images), labels
    )
)

test_dataset = test_dataset.map(
    lambda images, labels: (
        normalization_layer(images), labels
    )
)

# Optimize dataset loading
AUTOTUNE = 1

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    buffer_size=AUTOTUNE
)

# Display dataset information
print("\n" + "=" * 50)
print("       BLOOMSCAN DATA LOADER")
print("=" * 50)

print("\nFlower Classes:")
for index, flower in enumerate(FLOWER_CLASSES):
    print(f"{index}: {flower}")

print("\nImage Size:", IMAGE_SIZE)
print("Batch Size:", BATCH_SIZE)

print("\nTraining Batches:", len(train_dataset))
print("Validation Batches:", len(validation_dataset))
print("Testing Batches:", len(test_dataset))

print("\nData loading completed successfully!")
