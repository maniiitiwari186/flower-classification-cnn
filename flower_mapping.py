
# BloomScan - Oxford Flowers 102 Label Mapping

# Oxford dataset labels start from 1.

OXFORD_LABEL_TO_FLOWER = {
    2: "Orchid",
    6: "Lily",
    7: "Orchid",
    36: "Orchid",
    41: "Daisy",
    42: "Daffodil",
    49: "Daisy",
    50: "Dandelion",
    54: "Sunflower",
    67: "Crocus",
    74: "Rose",
    78: "Lotus",
}

# Oxford flower categories mapped to BloomScan
OXFORD_FLOWERS = sorted(set(OXFORD_LABEL_TO_FLOWER.values()))

# TensorFlow Flowers dataset categories
TENSORFLOW_FLOWERS = [
    "Daisy",
    "Dandelion",
    "Rose",
    "Sunflower",
    "Tulip",
]

# Final BloomScan classes
FLOWER_CLASSES = [
    "Rose",
    "Sunflower",
    "Daisy",
    "Tulip",
    "Lotus",
    "Dandelion",
    "Lily",
    "Orchid",
    "Daffodil",
    "Crocus",
]

if __name__ == "__main__":
    print("BloomScan Flower Mapping")
    print("------------------------")
    print("Total project classes:", len(FLOWER_CLASSES))
    print("Project classes:", FLOWER_CLASSES)
    print("Oxford mapped labels:", len(OXFORD_LABEL_TO_FLOWER))
    print("Oxford mapped categories:", OXFORD_FLOWERS)
