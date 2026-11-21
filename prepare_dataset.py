
from pathlib import Path
from collections import Counter
from scipy.io import loadmat

from flower_mapping import OXFORD_LABEL_TO_FLOWER, FLOWER_CLASSES


# BloomScan - Dataset Audit and Preparation

PROJECT_DIR = Path(__file__).resolve().parent

OXFORD_IMAGE_DIR = (
    PROJECT_DIR / "dataset" / "raw" / "oxford_flowers102" / "jpg"
)

OXFORD_LABEL_FILE = (
    PROJECT_DIR / "dataset" / "raw" / "oxford_flowers102" / "imagelabels.mat"
)

TF_IMAGE_DIR = PROJECT_DIR / "dataset" / "raw" / "flower_photos"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def count_images(folder):
    """Count image files inside a folder and its subfolders."""
    if not folder.exists():
        return 0

    return sum(
        1 for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def audit_oxford_dataset():
    """Count Oxford images belonging to our mapped flower classes."""

    labels = loadmat(OXFORD_LABEL_FILE)["labels"].ravel()

    counts = Counter()

    for index, label in enumerate(labels):
        flower_name = OXFORD_LABEL_TO_FLOWER.get(int(label))

        if flower_name is None:
            continue

        image_name = f"image_{index + 1:05d}.jpg"
        image_path = OXFORD_IMAGE_DIR / image_name

        if image_path.exists():
            counts[flower_name] += 1

    return counts


def audit_tensorflow_dataset():
    """Count images in the five TensorFlow Flowers folders."""

    folder_mapping = {
        "daisy": "Daisy",
        "dandelion": "Dandelion",
        "roses": "Rose",
        "sunflowers": "Sunflower",
        "tulips": "Tulip",
    }

    counts = Counter()

    for folder_name, flower_name in folder_mapping.items():
        folder = TF_IMAGE_DIR / folder_name
        counts[flower_name] = count_images(folder)

    return counts


def main():
    print("=" * 45)
    print("       BLOOMSCAN DATASET AUDIT")
    print("=" * 45)

    if not OXFORD_LABEL_FILE.exists():
        print("ERROR: Oxford label file not found!")
        return

    if not OXFORD_IMAGE_DIR.exists():
        print("ERROR: Oxford image folder not found!")
        return

    if not TF_IMAGE_DIR.exists():
        print("ERROR: TensorFlow flower folder not found!")
        return

    oxford_counts = audit_oxford_dataset()
    tensorflow_counts = audit_tensorflow_dataset()

    print("\nFLOWER-WISE IMAGE COUNTS")
    print("-" * 45)

    for flower in FLOWER_CLASSES:
        oxford = oxford_counts[flower]
        tensorflow = tensorflow_counts[flower]
        total = oxford + tensorflow

        print(
            f"{flower:18} | "
            f"Oxford: {oxford:4} | "
            f"TensorFlow: {tensorflow:4} | "
            f"Total: {total:4}"
        )

    print("\n" + "-" * 45)

    missing = [
        flower for flower in FLOWER_CLASSES
        if oxford_counts[flower] + tensorflow_counts[flower] == 0
    ]

    print("Total project classes:", len(FLOWER_CLASSES))
    print("Mapped Oxford labels:", len(OXFORD_LABEL_TO_FLOWER))
    print("Classes currently missing:", len(missing))

    if missing:
        print("\nMISSING FLOWER CATEGORIES:")
        for flower in missing:
            print("-", flower)
    else:
        print("\nAll flower categories have images.")

    print("\nOriginal datasets have NOT been modified.")
    print("Dataset audit completed!")


if __name__ == "__main__":
    main()
