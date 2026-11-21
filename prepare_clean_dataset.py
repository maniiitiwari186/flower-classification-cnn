
from pathlib import Path
import hashlib
import shutil

from scipy.io import loadmat

from flower_mapping import (
    FLOWER_CLASSES,
    OXFORD_LABEL_TO_FLOWER
)


# Dataset paths
BASE_DIR = Path("dataset/raw")

TENSORFLOW_DIR = BASE_DIR / "flower_photos"
OXFORD_DIR = BASE_DIR / "oxford_flowers102"
OXFORD_IMAGES_DIR = OXFORD_DIR / "jpg"

OUTPUT_DIR = Path("dataset/processed_v2")


# Calculate exact image hash
def calculate_hash(image_path):
    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# Create all 15 flower folders
for flower in FLOWER_CLASSES:
    (OUTPUT_DIR / flower).mkdir(parents=True, exist_ok=True)


# Track copied images and duplicate hashes
seen_hashes = set()
image_counts = {flower: 0 for flower in FLOWER_CLASSES}

total_duplicates = 0
total_copied = 0


def copy_image(source_path, flower):
    global total_duplicates, total_copied

    try:
        image_hash = calculate_hash(source_path)

        if image_hash in seen_hashes:
            total_duplicates += 1
            return

        destination_folder = OUTPUT_DIR / flower
        destination_path = destination_folder / source_path.name

        # Avoid filename conflicts
        if destination_path.exists():
            destination_path = destination_folder / (
                source_path.stem + "_copy" + source_path.suffix
            )

        shutil.copy2(source_path, destination_path)

        seen_hashes.add(image_hash)
        image_counts[flower] += 1
        total_copied += 1

    except Exception as error:
        print(f"Could not copy: {source_path}")
        print(error)


# 1. Copy TensorFlow Flowers images
print("Processing TensorFlow Flowers Dataset...")

tensorflow_folders = {
    "daisy": "Daisy",
    "dandelion": "Dandelion",
    "roses": "Rose",
    "sunflowers": "Sunflower",
    "tulips": "Tulip"
}

for folder, flower in tensorflow_folders.items():

    folder_path = TENSORFLOW_DIR / folder

    for image_path in folder_path.iterdir():

        if image_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            copy_image(image_path, flower)


# 2. Copy selected Oxford Flowers images
print("Processing Oxford Flowers 102 Dataset...")

labels = loadmat(OXFORD_DIR / "imagelabels.mat")["labels"].ravel()

for index, label in enumerate(labels, start=1):

    flower = OXFORD_LABEL_TO_FLOWER.get(int(label))

    if flower is None:
        continue

    image_path = OXFORD_IMAGES_DIR / f"image_{index:05d}.jpg"

    if image_path.exists():
        copy_image(image_path, flower)


# 3. Final report
print("\n" + "=" * 55)
print("       BLOOMSCAN CLEAN DATASET REPORT")
print("=" * 55)

for flower in FLOWER_CLASSES:
    print(f"{flower:<18} | {image_counts[flower]:>5} images")

print("-" * 55)
print(f"Total copied images: {total_copied}")
print(f"Duplicate images skipped: {total_duplicates}")
print(f"Total flower categories: {len(FLOWER_CLASSES)}")

print("\nOriginal datasets have NOT been modified.")
print("Clean dataset preparation completed!")
