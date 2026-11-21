
from pathlib import Path
from collections import Counter
from PIL import Image
from scipy.io import loadmat

from flower_mapping import OXFORD_LABEL_TO_FLOWER, FLOWER_CLASSES


# Dataset locations
BASE_DIR = Path("dataset/raw")
TENSORFLOW_DIR = BASE_DIR / "flower_photos"
OXFORD_DIR = BASE_DIR / "oxford_flowers102"
OXFORD_IMAGES_DIR = OXFORD_DIR / "jpg"


# Count image files and check their readability
def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()

        with Image.open(path) as img:
            img.load()

        return True

    except Exception:
        return False


results = {
    "TensorFlow": Counter(),
    "Oxford": Counter()
}

corrupt_images = []
checked_images = 0


print("=" * 55)
print("       BLOOMSCAN IMAGE QUALITY AUDIT")
print("=" * 55)


# 1. Check TensorFlow Flowers images
print("\nChecking TensorFlow Flowers Dataset...")

tensorflow_folders = {
    "daisy": "Daisy",
    "dandelion": "Dandelion",
    "roses": "Rose",
    "sunflowers": "Sunflower",
    "tulips": "Tulip"
}

for folder, flower in tensorflow_folders.items():
    folder_path = TENSORFLOW_DIR / folder

    for path in folder_path.glob("*"):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        checked_images += 1

        if check_image(path):
            results["TensorFlow"][flower] += 1
        else:
            corrupt_images.append(str(path))


# 2. Check Oxford Flowers 102 images
print("Checking Oxford Flowers 102 Dataset...")

mat_data = loadmat(OXFORD_DIR / "imagelabels.mat")
labels = mat_data["labels"].ravel()

for index, label in enumerate(labels, start=1):

    flower = OXFORD_LABEL_TO_FLOWER.get(int(label))

    if flower is None:
        continue

    image_path = OXFORD_IMAGES_DIR / f"image_{index:05d}.jpg"

    checked_images += 1

    if not image_path.exists():
        corrupt_images.append(str(image_path))
        continue

    if check_image(image_path):
        results["Oxford"][flower] += 1
    else:
        corrupt_images.append(str(image_path))


# 3. Print final report
print("\n" + "=" * 55)
print("              QUALITY AUDIT REPORT")
print("=" * 55)

print(f"\nTotal images checked: {checked_images}")

print("\nFLOWER-WISE VALID IMAGE COUNTS")
print("-" * 55)

for flower in FLOWER_CLASSES:
    tf_count = results["TensorFlow"][flower]
    ox_count = results["Oxford"][flower]
    total = tf_count + ox_count

    print(
        f"{flower:<18} | "
        f"TensorFlow: {tf_count:>4} | "
        f"Oxford: {ox_count:>4} | "
        f"Total: {total:>4}"
    )


print("\n" + "-" * 55)
print(f"Total valid images: {sum(results['TensorFlow'].values()) + sum(results['Oxford'].values())}")
print(f"Corrupt or missing images: {len(corrupt_images)}")

if corrupt_images:
    print("\nCORRUPT OR MISSING IMAGE PATHS:")
    for path in corrupt_images:
        print(path)
else:
    print("\nAll checked images are readable!")

print("\nImage quality audit completed!")
