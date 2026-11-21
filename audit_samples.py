
from pathlib import Path
import random

from PIL import Image, ImageOps, ImageDraw, ImageFont

BASE_DIR = Path("dataset/processed_v2")
OUTPUT_DIR = Path("audit_reports")

FLOWERS = ["Daisy", "Tulip"]

SAMPLES_PER_CLASS = 20
COLUMNS = 4

CELL_WIDTH = 240
CELL_HEIGHT = 205

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rng = random.Random(42)

for flower in FLOWERS:

    folder = BASE_DIR / flower

    images = sorted([
        path for path in folder.iterdir()
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    samples = rng.sample(
        images,
        min(SAMPLES_PER_CLASS, len(images))
    )

    rows = (len(samples) + COLUMNS - 1) // COLUMNS

    sheet = Image.new(
        "RGB",
        (COLUMNS * CELL_WIDTH, rows * CELL_HEIGHT),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(samples):

        try:
            image = Image.open(image_path).convert("RGB")

            image = ImageOps.fit(
                image,
                (220, 155)
            )

            x = (index % COLUMNS) * CELL_WIDTH
            y = (index // COLUMNS) * CELL_HEIGHT

            sheet.paste(image, (x + 10, y + 35))

            filename = image_path.name

            draw.text(
                (x + 10, y + 8),
                f"{index + 1:02d}. {filename[:28]}",
                fill="black"
            )

        except Exception as error:
            print("Could not open:", image_path, error)

    output_path = OUTPUT_DIR / f"{flower.lower()}_samples.jpg"

    sheet.save(output_path, quality=90)

    print(f"\n{flower} contact sheet saved:")
    print(output_path)

print("\nDataset audit samples generated successfully!")
