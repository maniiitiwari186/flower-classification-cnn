

#import tensorflow as tf

#from tensorflow.keras import layers, models
#from flower_mapping import FLOWER_CLASSES

# Model configuration
#IMAGE_SIZE = (224, 224)
#NUM_CLASSES = len(FLOWER_CLASSES)

# Build CNN model
#def build_cnn_model():

 #   model = models.Sequential([

 #       # Input layer
 #       layers.Input(shape=(224, 224, 3)),

        # First Convolution Block
  #      layers.Conv2D(32, (3, 3), activation="relu"),
  #      layers.MaxPooling2D((2, 2)),

        # Second Convolution Block
 #       layers.Conv2D(64, (3, 3), activation="relu"),
 #       layers.MaxPooling2D((2, 2)),

        # Third Convolution Block
 #       layers.Conv2D(128, (3, 3), activation="relu"),
 #       layers.MaxPooling2D((2, 2)),

  #      # Fourth Convolution Block
 #       layers.Conv2D(128, (3, 3), activation="relu"),
  #      layers.MaxPooling2D((2, 2)),

        # Flatten layer
   #     layers.Flatten(),

        # Fully Connected Layer
   #     layers.Dense(256, activation="relu"),

        # Dropout to reduce overfitting
   #     layers.Dropout(0.5),

        # Output Layer
 #       layers.Dense(NUM_CLASSES, activation="softmax")
 #   ])

    # Compile model
 #   model.compile(
  #      optimizer="adam",
  #      loss="sparse_categorical_crossentropy",
  #      metrics=["accuracy"]
  #  )

 #   return model
#

#if __name__ == "__main__":

 #   cnn_model = build_cnn_model()

 #   print("\n" + "=" * 55)
 #   print("       BLOOMSCAN CNN MODEL")
 #   print("=" * 55)

  #  cnn_model.summary()

  #  print("\nTotal Flower Categories:", NUM_CLASSES)
  #  print("CNN model created successfully!")


import tensorflow as tf

from tensorflow.keras import layers, models
from flower_mapping import FLOWER_CLASSES

IMAGE_SIZE = (160 , 160)
NUM_CLASSES = len(FLOWER_CLASSES)


def build_cnn_model():

    model = models.Sequential([

        layers.Input(shape=(160 , 160 , 3)),

        # Data Augmentation
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),

        # First Convolution Block
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Second Convolution Block
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Third Convolution Block
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Fourth Convolution Block
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),

        # Feature Reduction
        layers.GlobalAveragePooling2D(),

        # Classification Layers
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),

        # Output Layer
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    cnn_model = build_cnn_model()

    print("\n" + "=" * 55)
    print("       BLOOMSCAN OPTIMIZED CNN MODEL")
    print("=" * 55)

    cnn_model.summary()

    print("\nTotal Flower Categories:", NUM_CLASSES)
    print("Optimized CNN model created successfully!")
