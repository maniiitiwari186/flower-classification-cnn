
import tensorflow as tf
import numpy as np

from data_loader import test_dataset
from flower_mapping import FLOWER_CLASSES

# Load trained BloomScan V2 model
model = tf.keras.models.load_model(
    "models/bloomscan_v2_best.keras"
)

# Store actual and predicted labels
actual_labels = []
predicted_labels = []

print("\nEvaluating BloomScan V2...")

for images, labels in test_dataset:

    predictions = model.predict(images, verbose=0)

    predicted = np.argmax(predictions, axis=1)

    actual_labels.extend(labels.numpy())
    predicted_labels.extend(predicted)

actual_labels = np.array(actual_labels)
predicted_labels = np.array(predicted_labels)

# Calculate class-wise performance
print("\n" + "=" * 65)
print("       BLOOMSCAN CLASS-WISE EVALUATION")
print("=" * 65)

print(f"\n{'Flower':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
print("-" * 50)

for index, flower in enumerate(FLOWER_CLASSES):

    mask = actual_labels == index

    total = np.sum(mask)
    correct = np.sum(predicted_labels[mask] == index)

    accuracy = (correct / total * 100) if total > 0 else 0

    print(
        f"{flower:<15} {correct:<10} "
        f"{total:<10} {accuracy:.2f}%"
    )

# Overall accuracy
overall_accuracy = np.mean(
    actual_labels == predicted_labels
) * 100

print("\n" + "=" * 65)
print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
print("=" * 65)

print("\nEvaluation completed successfully!")
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(
    actual_labels,
    predicted_labels,
    labels=range(len(FLOWER_CLASSES))
)

print("\nCONFUSION MATRIX")
print("-" * 65)

print(f"{'Actual / Predicted':<18}", end="")

for flower in FLOWER_CLASSES:
    print(f"{flower[:8]:>9}", end="")

print()

for index, flower in enumerate(FLOWER_CLASSES):
    print(f"{flower:<18}", end="")

    for value in cm[index]:
        print(f"{value:>9}", end="")

    print()
print("\nPREDICTION DISTRIBUTION")
print("-" * 40)

for index, flower in enumerate(FLOWER_CLASSES):

    count = np.sum(predicted_labels == index)

    print(f"{flower:<15} {count:>5} predictions")