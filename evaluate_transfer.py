
import tensorflow as tf
import numpy as np

from data_loader import test_dataset
from flower_mapping import FLOWER_CLASSES

MODEL_PATH = "models/bloomscan_transfer_fixed_best.keras"

print("Loading BloomScan Transfer Model...")

model = tf.keras.models.load_model(MODEL_PATH)

correct = np.zeros(len(FLOWER_CLASSES), dtype=np.int64)
total = np.zeros(len(FLOWER_CLASSES), dtype=np.int64)

confusion = np.zeros(
    (len(FLOWER_CLASSES), len(FLOWER_CLASSES)),
    dtype=np.int64
)

print("\nEvaluating test dataset...\n")

for images, labels in test_dataset:
    predictions = model(images, training=False).numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    actual_labels = labels.numpy()

    for actual, predicted in zip(actual_labels, predicted_labels):
        total[actual] += 1
        confusion[actual][predicted] += 1

        if actual == predicted:
            correct[actual] += 1

overall_accuracy = correct.sum() / total.sum() * 100

print("=" * 65)
print("       BLOOMSCAN TRANSFER LEARNING EVALUATION")
print("=" * 65)

print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%\n")

print(f"{'Flower':<15}{'Correct':<12}{'Total':<10}{'Accuracy'}")
print("-" * 55)

for index, flower in enumerate(FLOWER_CLASSES):
    accuracy = (
        correct[index] / total[index] * 100
        if total[index] > 0 else 0
    )

    print(
        f"{flower:<15}{correct[index]:<12}"
        f"{total[index]:<10}{accuracy:.2f}%"
    )

print("\nCONFUSION MATRIX")
print("Rows = Actual | Columns = Predicted\n")

print(f"{'':<12}", end="")
for flower in FLOWER_CLASSES:
    print(f"{flower[:7]:>9}", end="")
print()

for index, flower in enumerate(FLOWER_CLASSES):
    print(f"{flower:<12}", end="")
    for value in confusion[index]:
        print(f"{value:>9}", end="")
    print()

print("\nEvaluation completed successfully!")
