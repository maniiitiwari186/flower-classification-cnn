
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

from flower_mapping import FLOWER_CLASSES

MODEL_PATH = "models/bloomscan_transfer_fixed_best.keras"
IMAGE_SIZE = (160, 160)


def predict_flower(image_path):

    # Check image file
    image_path = Path(image_path)

    if not image_path.exists():
        print("Error: Image file not found!")
        return

    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict flower
    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)
    flower_name = FLOWER_CLASSES[predicted_index]
    confidence = predictions[predicted_index] * 100

    print("\n" + "=" * 40)
    print("          BLOOMSCAN RESULT")
    print("=" * 40)

    print("Flower Name :", flower_name)
    print(f"Confidence  : {confidence:.2f}%")

    print("=" * 40)

    return flower_name, confidence


if __name__ == "__main__":

    image_folder = Path("dataset/processed_v2/Rose")

    images = [
        file for file in image_folder.iterdir()
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if images:
        image_path = images[0]

        print("\nAutomatically selected image:")
        print(image_path)

        predict_flower(image_path)

    else:
        print("No flower images found!")