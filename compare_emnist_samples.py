import os
import pandas as pd
import numpy as np
from PIL import Image

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "emnist_data")
TRAIN_CSV = os.path.join(DATA_DIR, "emnist-balanced-train.csv")

print(f"Loading training data from {TRAIN_CSV} ...")
train_df = pd.read_csv(TRAIN_CSV, header=None)

# Get first 4 rows where label == 7
seven_rows = train_df[train_df[0] == 7].head(4)
print(f"Found {len(seven_rows)} samples of label 7")

for idx, row in enumerate(seven_rows.itertuples(index=False), start=1):
    label = row[0]
    pixel_vals = np.array(row[1:], dtype=np.uint8)
    # reshape to (1,28,28,1)
    img = pixel_vals.reshape(1, 28, 28, 1)
    # transpose to fix EMNIST orientation
    img = np.transpose(img, (0, 2, 1, 3))
    # normalize
    img = img.astype(np.float32) / 255.0
    # back to uint8 for saving
    img_uint8 = (img[0, :, :, 0] * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode="L")
    # upscale to 280x280 nearest neighbor
    pil_img = pil_img.resize((280, 280), Image.NEAREST)
    out_path = os.path.join(BASE_DIR, f"emnist_sample_7_{idx}.png")
    pil_img.save(out_path)
    print(f"Saved sample {idx} to {out_path}")

print("All done.")
