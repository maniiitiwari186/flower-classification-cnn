
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from flower_mapping import FLOWER_CLASSES

IMAGE_SIZE = (160, 160)
NUM_CLASSES = len(FLOWER_CLASSES)

def build_transfer_model():
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(160, 160, 3)
    )

    # Initially freeze the pretrained layers
    base_model.trainable = False

    inputs = layers.Input(shape=(160, 160, 3))

    # MobileNetV2 expects preprocessing in the range [-1, 1]
    x = layers.Rescaling(scale=2.0, offset=-1)(inputs)

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_transfer_model()
    model.summary()
    print("\nSUCCESS: BloomScan Transfer Learning Model Ready!")
