# ✍️ Handwritten Character Recognition

A web-based handwritten character recognition app built with a Convolutional Neural Network (CNN) trained on the **EMNIST Balanced** dataset. Draw any digit or letter on the canvas and get real-time predictions with confidence scores.

**CodeAlpha Internship Project**

---

## 🎯 Features

- 🖌️ Interactive drawing canvas — draw characters with your mouse or touchscreen
- 🧠 CNN model trained on EMNIST Balanced (47 classes: digits 0–9, uppercase A–Z, and select lowercase letters)
- 📊 Real-time prediction with top-3 confidence scores
- ⚡ Lightweight Flask backend serving a trained Keras model
- 🎨 Clean, responsive UI

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| Model | TensorFlow / Keras (CNN) |
| Dataset | EMNIST Balanced (47 classes) |
| Frontend | HTML, CSS, JavaScript (Canvas API) |
| Image Processing | Pillow (PIL), NumPy |

## 📂 Project Structure

```
CodeAlpha_HandwrittenCharacterRecognition/
├── app.py                  # Flask app — routes, preprocessing, prediction
├── model/
│   ├── character_model.keras   # Trained CNN model
│   └── label_mapping.pkl       # Class index → character mapping
├── templates/
│   └── index.html          # Frontend UI (drawing canvas + results)
├── requirements.txt         # Python dependencies
└── notebook/                # Model training notebook
```

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- pip

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/shoaibzoya8-png/CodeAlpha_HandwrittenCharacterRecognition.git
   cd CodeAlpha_HandwrittenCharacterRecognition
   ```

2. Create a virtual environment and activate it
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app
   ```bash
   python app.py
   ```

5. Open your browser and go to:
   ```
   http://127.0.0.1:5000
   ```

## 🧪 How It Works

1. The user draws a character on an HTML5 canvas.
2. On clicking **Predict**, the canvas image is sent to the Flask backend.
3. The backend preprocesses the image to match the EMNIST training format:
   - Converts to grayscale and thresholds to a pure black background
   - Detects the bounding box of the drawn strokes and crops to it
   - Centers and pads the character, then resizes to 28×28 using smooth interpolation
4. The preprocessed image is fed into the trained CNN model.
5. The top-3 predicted classes with confidence scores are returned and displayed.

## 📊 Model

The CNN was trained on the **EMNIST Balanced** dataset (47 balanced classes covering digits and letters). Preprocessing during inference closely mirrors the EMNIST format (centered, scaled, black-background characters) to maximize prediction accuracy on freehand canvas input.

## 📌 Future Improvements

- [ ] Support multi-character (word-level) recognition
- [ ] Add model confidence visualization/heatmap
- [ ] Deploy to a public hosting service (e.g., Render, Hugging Face Spaces)

## 🙋 Author

**Zoya Shoaib**
CodeAlpha Internship — Machine Learning Track

## 📄 License

This project is open-source and available for educational purposes.
