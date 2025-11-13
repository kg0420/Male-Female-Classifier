import os, uuid
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# -------------------
# Config
# -------------------
ALLOWED_EXT = {"jpg", "jpeg", "png"}
UPLOAD_DIR   = os.path.join("static", "uploads")
OUT_DIR      = os.path.join("static", "predictions")
IMG_SIZE     = (224, 224)  # EfficientNetB3

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "super-secret-key"  # change in prod

# -------------------
# Load model once
# -------------------
MODEL_PATH = "Male_Female_Detection_Model.h5"
model = load_model(MODEL_PATH)

# Match your class order used when training
CLASS_NAMES = ["female","male"]

# -------------------
# Helpers
# -------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def read_rgb(path: str):
    bgr = cv2.imread(path)
    if bgr is None:
        raise ValueError("Could not read uploaded image.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def grad_cam_overlay(img_rgb: np.ndarray, heatmap: np.ndarray, alpha=0.35):
    """Overlay CAM heatmap on RGB image (img 0..255)."""
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay_bgr = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 1 - alpha, heatmap_color, alpha, 0)
    return overlay_bgr

def compute_grad_cam(model, img_batch, last_conv_name="top_conv"):
    """
    Returns normalized Grad-CAM heatmap for the most confident class.
    img_batch: (1, H, W, 3) preprocessed
    """
    try:
        last_conv_layer = model.get_layer(last_conv_name)
    except Exception:
        # Try to guess the last conv layer if name changed
        conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        last_conv_layer = conv_layers[-1]

    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_batch, training=False)
        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)  # (1, h, w, c)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (c,)
    conv_out = conv_out[0]  # (h, w, c)

    # Weight the channels by the importance
    conv_out = conv_out * pooled_grads
    heatmap = tf.reduce_mean(conv_out, axis=-1)

    # Normalize 0..1
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def run_inference(img_path: str):
    """
    Returns: (pred_label:str, confidence:float, cam_path:str)
    """
    # Read and keep a display copy (RGB for overlay)
    rgb = read_rgb(img_path)
    disp_rgb = cv2.resize(rgb, (480, 480))  # nicer on page

    # Preprocess for EfficientNet
    in_img = cv2.resize(rgb, IMG_SIZE)
    in_arr = img_to_array(in_img)
    in_arr = preprocess_input(in_arr)
    in_arr = np.expand_dims(in_arr, axis=0)

    # Predict
    preds = model.predict(in_arr, verbose=0)[0]
    class_id = int(np.argmax(preds))
    label = CLASS_NAMES[class_id]
    conf = float(preds[class_id] * 100.0)

    # Grad-CAM
    heatmap = compute_grad_cam(model, in_arr, last_conv_name="top_conv")
    overlay_bgr = grad_cam_overlay(disp_rgb, heatmap, alpha=0.38)

    # Save overlay
    out_name = f"{uuid.uuid4().hex}.png"
    out_path = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_path, overlay_bgr)

    return label, conf, out_name

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
            flash("Allowed types: jpg, jpeg, png")
            return redirect(url_for("index"))

        filename = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
        f.save(save_path)

        try:
            label, conf, cam_name = run_inference(save_path)
            return render_template(
                "index.html",
                pred_label=label,
                pred_conf=f"{conf:.2f}",
                cam_url=url_for("static", filename=f"predictions/{cam_name}"),
                uploaded_url=url_for("static", filename=f"uploads/{os.path.basename(save_path)}"),
            )
        except Exception as e:
            print("Inference error:", e)
            flash(f"Inference error: {e}")
            return redirect(url_for("index"))

    return render_template("index.html", pred_label=None)

if __name__ == "__main__":
    app.run(debug=True)
