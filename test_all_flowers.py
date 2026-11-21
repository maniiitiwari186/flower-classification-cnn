
import tensorflow as tf
import numpy as np
from pathlib import Path
from PIL import Image

from flower_mapping import FLOWER_CLASSES

MODEL_PATH = "models/bloomscan_transfer_fixed_best.keras"
DATASET_DIR = Path("dataset/processed_v2")
IMAGE_SIZE = (160, 160)

print("Loading BloomScan model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("\n" + "=" * 65)
print("             BLOOMSCAN - 10 FLOWER TEST")
print("=" * 65)

for flower in FLOWER_CLASSES:

    folder = DATASET_DIR / flower

    images = sorted([
        file for file in folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if not images:
        print(f"{flower:<15} | No image found")
        continue

    image = Image.open(images[0]).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    predicted_flower = FLOWER_CLASSES[predicted_index]
    confidence = predictions[predicted_index] * 100

    status = "PASS" if predicted_flower == flower else "CHECK"

    print(
        f"{flower:<15} | Predicted: {predicted_flower:<12} "
        f"| {confidence:6.2f}% | {status}"
    )

print("\n10 Flower Testing Completed!")
