import base64
import io
import json
import os
import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
import tensorflow as tf

# ── App & Model Setup ─────────────────────────────────────────────────────────
app = Flask(__name__)

MODEL_PATH = "model/character_model.keras"
LABEL_MAP_PATH = "model/label_mapping.pkl"

print("Loading model…")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "rb") as f:
    label_mapping = pickle.load(f)  # {0: '0', 1: '1', …, 46: 'Z'}

print(f"Model loaded. Classes: {len(label_mapping)}")

# ── Image Pre-processing ──────────────────────────────────────────────────────
def preprocess_canvas_image(b64_string: str) -> np.ndarray:
    """Preprocess a base64‑encoded canvas image for the EMNIST model.

    Steps:
    1. Strip data‑URI header.
    2. Decode PNG, composite onto black background, convert to grayscale.
    3. Auto‑invert if background is light.
    4. Detect bounding box of the drawing, add 4 px padding, crop.
    5. Resize cropped region to 20×20, paste onto a 28×28 black canvas centered.
    6. Convert to float32, min‑max normalize, reshape to (1, 28, 28, 1).
    """
    # 1. Strip header
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]

    # 2. Decode to PIL image
    img_bytes = base64.b64decode(b64_string)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # 3. Composite onto black background, then grayscale
    background = Image.new("RGBA", img.size, (0, 0, 0, 255))
    background.paste(img, mask=img.split()[3])
    img_gray = background.convert("L")

    # 4. Auto‑invert if background appears light
    if np.array(img_gray).mean() > 127:
        img_gray = ImageOps.invert(img_gray)

    # NEW: Threshold to pure black/white to remove gray artifacts
    img_gray = img_gray.point(lambda p: 255 if p > 127 else 0)

    # 5. Bounding‑box detection with 4 px padding
    arr_np = np.array(img_gray)
    rows = np.where(arr_np.max(axis=1) > 0)[0]
    cols = np.where(arr_np.max(axis=0) > 0)[0]
    if rows.size == 0 or cols.size == 0:
        # Empty drawing – plain black 28×28 image
        img_centered = Image.new("L", (28, 28), 0)
    else:
        top, bottom = rows[0], rows[-1]
        left, right = cols[0], cols[-1]
        pad = 2  # reduced padding to ensure digit occupies more of the 28×28 canvas
        top = max(top - pad, 0)
        left = max(left - pad, 0)
        bottom = min(bottom + pad, img_gray.height - 1)
        right = min(right + pad, img_gray.width - 1)
        img_cropped = img_gray.crop((left, top, right + 1, bottom + 1))
        img_resized20 = img_cropped.resize((20, 20), Image.LANCZOS)
        img_centered = Image.new("L", (28, 28), 0)
        img_centered.paste(img_resized20, (4, 4))

    # 6. Convert to float32 NumPy array
    arr = np.array(img_centered, dtype=np.float32)

    # 7. Min‑max normalization
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)

    # 8. Add batch & channel dimensions
    arr = arr.reshape(1, 28, 28, 1)
    return arr

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data or "image" not in data:
        return jsonify({"error": "No image data received"}), 400
    try:
        img_array = preprocess_canvas_image(data["image"])
        # Save debug image (28x28 scaled to 280x280 for inspection)
        debug_arr = img_array[0, :, :, 0]
        debug_img_data = (debug_arr * 255.0).astype(np.uint8)
        debug_img = Image.fromarray(debug_img_data, mode="L")
        debug_img_resized = debug_img.resize((280, 280), Image.NEAREST)
        debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_preprocessed.png")
        debug_img_resized.save(debug_path)
        print(f"DEBUG LIVE PREDICT STATS: min={img_array.min():.4f}, max={img_array.max():.4f}, mean={img_array.mean():.4f}")
        import sys
        sys.stdout.flush()
    except Exception as exc:
        return jsonify({"error": f"Image processing failed: {exc}"}), 422

    # Run inference
    preds = model.predict(img_array, verbose=0)[0]

    # Top‑3 predictions
    top3_indices = np.argsort(preds)[::-1][:3]
    results = [
        {
            "character": label_mapping[int(idx)],
            "confidence": round(float(preds[idx]) * 100, 2),
        }
        for idx in top3_indices
    ]
    return jsonify({"predictions": results})

# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
