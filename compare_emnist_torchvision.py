import os
import torch
from torchvision import datasets
import numpy as np
from PIL import Image

# Directory to store the EMNIST download
data_dir = os.path.join(os.getcwd(), 'emnist_torch')
os.makedirs(data_dir, exist_ok=True)

# Download EMNIST balanced split (training set)
emnist = datasets.EMNIST(root=data_dir, split='balanced', download=True, train=True)

# Find indices where the label is 7 (digit 7 in the balanced set)
indices = (emnist.targets == 7).nonzero(as_tuple=False).squeeze()
selected = indices[:4]  # take first 4 samples (or fewer if not enough)

for i, idx in enumerate(selected, start=1):
    # emnist.data is a tensor of shape (N, 28, 28) with uint8 values 0‑255
    img_array = emnist.data[idx].numpy()  # (28, 28)
    # Reshape to (1, 28, 28, 1) to match the preprocessing pipeline used in training
    img = img_array.reshape(1, 28, 28, 1)
    # Apply the same transpose that the notebook uses: (0, 2, 1, 3)
    img = np.transpose(img, (0, 2, 1, 3))
    # Normalize pixel values to [0, 1]
    img = img.astype(np.float32) / 255.0
    # Convert back to uint8 for saving as a PNG
    img_uint8 = (img[0, :, :, 0] * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8, mode='L')
    # Upscale to 280x280 using nearest‑neighbor (same as debug_preprocessed.png)
    pil_img = pil_img.resize((280, 280), Image.NEAREST)
    out_path = os.path.join(os.getcwd(), f'emnist_torch_7_{i}.png')
    pil_img.save(out_path)
    print(f'Saved sample {i} to {out_path}')

print('All done.')
