<div align="center">

# 😷 FONet — FacialOcclusionNet
### Real-Time Face Mask Detection using Deep Learning

[![Accuracy](https://img.shields.io/badge/Accuracy-99.34%25-brightgreen?style=for-the-badge)](https://github.com/DX-STAR/Research-Paper----FONET-Model)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

> *A research-backed AI system that can tell — in real time — whether a person is wearing a face mask or not, just by looking through a camera.*

</div>

---

## 🧠 What Is This Project? (For Everyone)

Imagine standing at the entrance of an airport or a shopping mall during a health crisis like COVID-19. A security guard would normally have to check every single person to make sure they are wearing a mask — which is slow, tiring, and sometimes unsafe.

**This project solves that problem with Artificial Intelligence (AI).**

We built a system called **FONet (FacialOcclusionNet)** — a smart computer program that:

- 📷 **Looks at a live camera feed**
- 🔍 **Detects every face in the frame**
- ✅ **Instantly tells** if the person is wearing a mask or not
- 🟢 Shows a **green box** around masked faces
- 🔴 Shows a **red box** around unmasked faces

All of this happens **in real time** — faster than a human blink.

---

## 🎯 Why Does This Matter?

During the **COVID-19 pandemic**, wearing face masks became essential to protect public health. But enforcing this rule manually in crowded places like:

- 🛫 **Airports**
- 🚉 **Railway Stations**
- 🛒 **Shopping Malls**
- 🏥 **Hospitals**

...was nearly impossible to do at scale. Our AI system can **automate this process 24/7, without fatigue, and at very high accuracy.**

---

## 🏆 How Good Is It?

We trained and tested **two different AI models** and compared them:

| Model | What It Is | Accuracy |
|-------|-----------|----------|
| **Custom CNN** | A neural network we built from scratch | 96.69% |
| **FONet (Ours)** | Our improved model based on MobileNet | **99.34%** ✅ |

> 💡 **In plain English:** Out of every 100 faces it sees, FONet correctly identifies **99 of them**. That's better than most human-level performance in similar tasks.

Our model also has **better Precision, Recall, and F1 Score** — which means it makes fewer mistakes in both directions (rarely says "mask" when there isn't one, and rarely misses a mask-less face).

---

## 📊 The Data

We trained the model on a dataset of **7,553 real face images** — split into two categories:

```
📁 Dataset
├── 😷 With Mask    →  People wearing face masks
└── 😐 Without Mask →  People not wearing face masks
```

To make the model more robust (able to work in the real world), we used **Data Augmentation** — a technique where we artificially create variations of training images (rotating them, flipping them, adjusting brightness, etc.) so the model learns to recognize masks under different conditions.

---

## 🔬 How Does the AI Work? (Simple Explanation)

Think of the AI like training a very young child to recognize masks:

1. **Show it thousands of photos** — some with masks, some without
2. **The child learns patterns** — what a masked face "looks like" vs an unmasked one
3. **Test it on new photos** — ones it has never seen before
4. **It gets better over time** — this is called "training"

In technical terms, we used a technique called **Transfer Learning** — we took a pre-trained model called **MobileNet** (already smart at recognizing images) and fine-tuned it specifically for face mask detection.

```
Camera Feed
     ↓
Face Detection (OpenCV Haar Cascade)
     ↓
Face Image (150×150 pixels)
     ↓
FONet Model (Deep Learning)
     ↓
"Mask" ✅  or  "No Mask" ❌
```

---

## 🧰 What's Inside This Repository

```
📁 FONET Model/
├── 📓 FaceMask.ipynb       → The full code: training, testing, and live webcam detection
├── 🤖 best_model.keras     → The trained AI model (ready to use, no training needed!)
└── 📄 README.md            → You are here!
```

---

## 🚀 How to Run This Yourself

### Step 1 — Install Requirements
```bash
pip install tensorflow opencv-python numpy
```

### Step 2 — Open the Notebook
```bash
jupyter notebook FaceMask.ipynb
```

### Step 3 — Run the Live Detection Cell
The last cell in the notebook starts your **webcam** and runs real-time detection. Press `Q` to quit.

### Load the Model Directly (Advanced)
```python
import tensorflow as tf
import cv2
import numpy as np

# Load the model
model = tf.keras.models.load_model("best_model.keras")

# Feed an image (150x150 RGB)
img = cv2.resize(your_face_image, (150, 150)) / 255.0
prediction = model.predict(np.expand_dims(img, axis=0))[0][0]

label = "Mask ✅" if prediction < 0.5 else "No Mask ❌"
print(label)
```

---

## ⚙️ Technical Specifications

| Property | Value |
|----------|-------|
| Framework | TensorFlow / Keras |
| Base Architecture | MobileNet (fine-tuned) |
| Input Size | 150 × 150 × 3 (RGB) |
| Output | Binary (Mask / No Mask) |
| Dataset Size | 7,553 images |
| Face Detector | OpenCV Haar Cascade |
| Best Accuracy | **99.34%** |
| Inference Speed | ~90–120 ms/frame |
| Language | Python 3.10 |

---

## 📋 Requirements

```
tensorflow >= 2.x
opencv-python >= 4.x
numpy >= 1.21
jupyter (for running the notebook)
```

---

## ⚠️ Known Limitations

Even with 99.34% accuracy, the model has some known challenges:

| Challenge | Description |
|-----------|-------------|
| 😷 Improper Masks | If a mask is worn below the nose or chin, it may be misclassified |
| 💡 Extreme Lighting | Very dark or overexposed environments reduce accuracy |
| 🙌 Occlusions | Hands covering the face may confuse the model |

---

## 🔭 Future Work

The research suggests these improvements for next-generation versions:

- 🔧 **Model Pruning** — Making the model even smaller and faster
- 🤖 **Transformer Models** — Using newer AI architectures for better accuracy
- 📱 **Edge Computing** — Running the model on low-power devices (Raspberry Pi, phones)
- 🧬 **Multi-modal Biometrics** — Combining mask detection with other health indicators

---

## 📄 Research Paper

This project is backed by a **peer-reviewed research paper** on:

> *"FacialOcclusionNet (FONet): A Lightweight Deep Learning Model for Real-Time Face Mask Detection"*

**Key Findings from the Paper:**
- FONet outperforms existing MobileNetV2-based models
- Incorporates **lightweight optimizations** and **enhanced feature extraction**
- Achieves superior real-time detection efficiency
- Reduces **false positives** significantly compared to baseline models
- Generalizes well across varying lighting, mask types, and occlusions

**Keywords:** Face Mask Detection · Deep Learning · FacialOcclusionNet · Real-time Detection · Computer Vision · CNN · MobileNet · Image Classification · Occlusion Handling · Data Augmentation

---

## 👤 Author

**DX-STAR**
- GitHub: [@DX-STAR](https://github.com/DX-STAR)
- Repository: [Research-Paper----FONET-Model](https://github.com/DX-STAR/Research-Paper----FONET-Model)
- Paper Link: [Research-Paper](https://drive.google.com/file/d/1cY9xM4dUz56bLLmi02Qi_FDpqMemkk6h/view?usp=drive_link)

---

## 📜 License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it with attribution.

---

<div align="center">

*Built with ❤️ to protect public health through AI*

⭐ **If this project helped you, please give it a star!** ⭐

</div>
