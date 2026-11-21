
from pathlib import Path
from collections import defaultdict
import hashlib

from scipy.io import loadmat
from flower_mapping import OXFORD_LABEL_TO_FLOWER


BASE_DIR = Path("dataset/raw")

TENSORFLOW_DIR = BASE_DIR / "flower_photos"
OXFORD_DIR = BASE_DIR / "oxford_flowers102"
OXFORD_IMAGES_DIR = OXFORD_DIR / "jpg"


def calculate_hash(image_path):
    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


image_hashes = defaultdict(list)
total_scanned = 0


print("=" * 55)
print("       BLOOMSCAN DUPLICATE IMAGE AUDIT")
print("=" * 55)


# 1. Scan TensorFlow Flowers
print("\nScanning TensorFlow Flowers Dataset...")

for folder in TENSORFLOW_DIR.iterdir():

    if not folder.is_dir():
        continue

    for image_path in folder.iterdir():

        if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            image_hash = calculate_hash(image_path)
            image_hashes[image_hash].append(str(image_path))
            total_scanned += 1

        except Exception as error:
            print(f"Could not read: {image_path}")
            print(error)


# 2. Scan mapped Oxford Flowers images
print("\nScanning Oxford Flowers 102 Dataset...")

labels = loadmat(OXFORD_DIR / "imagelabels.mat")["labels"].ravel()

for index, label in enumerate(labels, start=1):

    flower = OXFORD_LABEL_TO_FLOWER.get(int(label))

    if flower is None:
        continue

    image_path = OXFORD_IMAGES_DIR / f"image_{index:05d}.jpg"

    if not image_path.exists():
        continue

    try:
        image_hash = calculate_hash(image_path)
        image_hashes[image_hash].append(str(image_path))
        total_scanned += 1

    except Exception as error:
        print(f"Could not read: {image_path}")
        print(error)


# 3. Find exact duplicates
duplicate_groups = [
    paths for paths in image_hashes.values()
    if len(paths) > 1
]


duplicate_image_count = sum(
    len(paths) - 1 for paths in duplicate_groups
)


# 4. Print report
print("\n" + "=" * 55)
print("              DUPLICATE AUDIT REPORT")
print("=" * 55)

print(f"\nTotal images scanned: {total_scanned}")
print(f"Unique image files: {len(image_hashes)}")
print(f"Duplicate groups found: {len(duplicate_groups)}")
print(f"Extra duplicate copies: {duplicate_image_count}")


if duplicate_groups:

    print("\nDUPLICATE IMAGE DETAILS")
    print("-" * 55)

    for number, paths in enumerate(duplicate_groups, start=1):

        print(f"\nDuplicate Group {number}:")

        for path in paths:
            print(path)

else:
    print("\nNo exact duplicate images found!")


print("\nOriginal dataset files have NOT been modified.")
print("Duplicate audit completed!")
