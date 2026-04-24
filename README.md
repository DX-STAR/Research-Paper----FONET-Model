# FONET Model — Face Mask Detection Research

This repository contains the code and trained model for the **FONET (Face-Oriented Network)** model used for face mask detection.

## 📂 Contents

| File | Description |
|------|-------------|
| `FaceMask.ipynb` | Jupyter notebook with full training pipeline, evaluation, and results |
| `best_model.keras` | Best trained Keras model weights (~37.6 MB) |

## 🚀 Getting Started

### Requirements
- Python 3.8+
- TensorFlow / Keras
- Jupyter Notebook

### Run the Notebook
```bash
pip install tensorflow jupyter
jupyter notebook FaceMask.ipynb
```

### Load the Model
```python
from tensorflow import keras

model = keras.models.load_model("best_model.keras")
model.summary()
```

## 📄 Research

This model is part of an ongoing research paper on lightweight face mask detection using the FONET architecture.

---
*Maintained by [DX-STAR](https://github.com/DX-STAR)*
