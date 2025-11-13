import os, uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import onnxruntime as ort

# -------------------
# Config
# -------------------

ALLOWED_EXT = {"jpg", "jpeg", "png"}
UPLOAD_DIR = os.path.join("static", "uploads")
IMG_SIZE = (224, 224)

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "super-secret-key"

# -------------------
# Load ONNX Model
# -------------------

MODEL_PATH = "model.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASS_NAMES = ["female", "male"]


# -------------------
# Helpers
# -------------------

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image not readable")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    arr = img.astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def run_inference(img_path):
    arr = preprocess_image(img_path)

    preds = session.run([output_name], {input_name: arr})[0][0]

    class_id = int(np.argmax(preds))
    label = CLASS_NAMES[class_id]
    confidence = float(preds[class_id] * 100)

    return label, confidence


# -------------------
# Routes
# -------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        if "image" not in request.files:
            flash("Please choose an image.")
            return redirect(url_for("index"))

        f = request.files["image"]
        if f.filename == "":
            flash("No file selected.")
            return redirect(url_for("index"))

        if not allowed_file(f.filename):
            flash("Allowed formats: jpg, jpeg, png")
            return redirect(url_for("index"))

        filename = secure_filename(f.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)
        f.save(save_path)

        try:
            label, conf = run_inference(save_path)

            return render_template(
                "index.html",
                pred_label=label,
                pred_conf=f"{conf:.2f}",
                uploaded_url=url_for("static", filename=f"uploads/{unique_name}"),
                cam_url=url_for("static", filename=f"uploads/{unique_name}"),  # No CAM
            )

        except Exception as e:
            print("Inference error:", e)
            flash("Prediction error.")
            return redirect(url_for("index"))

    return render_template("index.html", pred_label=None)


if __name__ == "__main__":
    app.run(debug=True)
