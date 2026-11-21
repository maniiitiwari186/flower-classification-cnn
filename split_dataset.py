
from pathlib import Path
import random
import shutil

from flower_mapping import FLOWER_CLASSES

# Project directories
SOURCE_DIR = Path("dataset/processed_v2")
OUTPUT_DIR = Path("dataset/split_v2")
# Split ratio
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducible random splitting
random.seed(42)

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Create output folders
for split in ["train", "validation", "test"]:
    for flower in FLOWER_CLASSES:
        (OUTPUT_DIR / split / flower).mkdir(
            parents=True,
            exist_ok=True
        )

print("=" * 55)
print("       BLOOMSCAN DATASET SPLITTING")
print("=" * 55)

total_train = 0
total_validation = 0
total_test = 0

for flower in FLOWER_CLASSES:

    flower_folder = SOURCE_DIR / flower

    images = [
        image for image in flower_folder.iterdir()
        if image.is_file()
        and image.suffix.lower() in IMAGE_EXTENSIONS
    ]

    random.shuffle(images)

    total_images = len(images)

    train_count = int(total_images * TRAIN_RATIO)
    validation_count = int(total_images * VALIDATION_RATIO)

    train_images = images[:train_count]

    validation_images = images[
        train_count:train_count + validation_count
    ]

    test_images = images[
        train_count + validation_count:
    ]

    # Copy training images
    for image in train_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "train" / flower / image.name
        )

    # Copy validation images
    for image in validation_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "validation" / flower / image.name
        )

    # Copy testing images
    for image in test_images:
        shutil.copy2(
            image,
            OUTPUT_DIR / "test" / flower / image.name
        )

    total_train += len(train_images)
    total_validation += len(validation_images)
    total_test += len(test_images)

    print(f"\n{flower}")
    print(f"  Total images:      {total_images}")
    print(f"  Training:          {len(train_images)}")
    print(f"  Validation:        {len(validation_images)}")
    print(f"  Testing:           {len(test_images)}")

print("\n" + "=" * 55)
print("             FINAL DATASET REPORT")
print("=" * 55)

print(f"Training images:   {total_train}")
print(f"Validation images: {total_validation}")
print(f"Testing images:    {total_test}")
print(f"Total images:      {total_train + total_validation + total_test}")

print("\nDataset splitting completed!")
print("Original processed dataset has NOT been modified.")
