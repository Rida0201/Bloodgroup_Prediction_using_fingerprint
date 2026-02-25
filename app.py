from flask import Flask, render_template, request
import numpy as np
import cv2
from tensorflow import keras
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
);

# Load trained CNN model
model = keras.models.load_model("bloodgroup_cnn_model.h5")

# Class labels (same order as training)
classes = ["A", "B", "AB", "O"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image received"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    # Read image
    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_GRAYSCALE
    )

    # Preprocess
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = img.reshape(1, 128, 128, 1)

    # Predict
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    return render_template(
        "index.html",
        prediction=predicted_class,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)