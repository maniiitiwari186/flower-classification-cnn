from flask import Flask, request, jsonify
from PIL import Image
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/")
def home():
    return "Flower Classification CNN Backend is Running!"

@app.route("/upload", methods=["POST"])
def upload_image():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(file.stream)
        image.verify()

        file.stream.seek(0)

        filename = os.path.basename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        file.save(filepath)

        return jsonify({
            "message": "Image uploaded successfully",
            "filename": filename
        }), 200

    except Exception:
        return jsonify({"error": "Invalid image file"}), 400


if __name__ == "__main__":
    app.run(debug=True)