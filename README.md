# Document Tampering Detection & Localization

[![Course](https://img.shields.io/badge/Course-CSE310%20Neural%20Networks%20%26%20Deep%20Learning-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](#)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-orange.svg)](#)
[![License](https://img.shields.io/badge/Academic-Project-lightgrey.svg)](#)

An automated visual-forensic deep learning system designed to detect document forgery (splicing, text erasure, copy-move) and localize manipulated pixel regions across scanned documents (receipts, bills, financial statements, and ID forms).

Developed as the semester-end project for **CSE310: Neural Network and Deep Learning** at the Faculty of Engineering and Technology, South Asian University.

---

## 📌 Project Overview

Digital document fraud has become increasingly sophisticated across banking, insurance, and administrative sectors. Manual inspection of subtle micro-edits—such as altered invoice amounts, swapped dates, or forged stamps—is error-prone and time-consuming.

This project implements a multi-stage deep learning pipeline combining **Error Level Analysis (ELA)**, **Laplacian micro-edge filtering**, **Convolutional Neural Networks (CNNs)**, and **U-Net segmentation** to:
1. Classify documents as **Authentic** or **Tampered**.
2. Generate pixel-level **binary tampering masks** localizing the exact forged zones.
3. Provide visual explainability via **Grad-CAM** (Gradient-weighted Class Activation Mapping).

---

## 📊 Dataset: DocTamper

The project leverages the **DocTamper** benchmark dataset (120,000 scanned documents at 512×512 resolution):
* **Source & Structure:** LMDB database containing paired document images and binary ground-truth tampering masks (`0` = pristine background, `1` = forged pixel).
* **Key Observations from Exploratory Data Analysis (EDA):**
  * **Document-Level Imbalance:** ~84.6% tampered vs. 15.4% authentic documents due to synthetic generation of multiple manipulation variants per original document. Addressed via class-weighted loss and balanced sampling.
  * **Spatial Pixel Sparsity:** Over 90% of tampered documents alter less than 10% of total surface area. Addressed using region-weighted segmentation losses (Soft Dice Loss & Focal Loss).

---

## 🏗️ Proposed Methodology

```
Input Document (512×512)
       │
       ├──► RGB Visual Stream
       ├──► Error Level Analysis (ELA) Stream (90% quality, 25× multiplier)
       └──► Laplacian High-Pass Edge Filter
              │
              ├──► [Model 1] Dual-Stream CNN Classifier ──► Verdict (Authentic / Tampered)
              │                                      └──► Grad-CAM Interpretability Map
              │
              ├──► [Model 2] U-Net Segmentation Network ──► Pixel Tampering Mask
              │
              └──► [Model 3] Denoising Autoencoder       ──► Anomaly Reconstruction Residual
```

### Course Syllabus Alignment (CSE310)
* **Module 1 (Foundations):** Binary Cross-Entropy loss, confusion matrices, Precision, Recall, Macro-F1, and ROC-AUC curves.
* **Module 2 (Optimization & Diagnostics):** Adam optimizer with learning rate warmup/decay schedules, early stopping, dropout, batch normalization, and class-weight balancing.
* **Module 3 (Spatial Data & Computer Vision):** Transfer learning (MobileNetV2 / ResNet), U-Net encoder-decoder with skip connections, and Grad-CAM interpretability.
* **Module 5 (Generative Models):** Unsupervised Denoising Autoencoder for patch anomaly detection.

---

## 🎯 Target Performance Metrics

| Task | Primary Metric | Target |
| :--- | :--- | :--- |
| **Document Classification** | Macro F1-Score | $\ge 0.90$ |
| **Document Classification** | AUC-ROC | $\ge 0.93$ |
| **Tampered Region Localization** | Intersection over Union (IoU) | $\ge 0.65$ |
| **Tampered Region Localization** | Dice Similarity Coefficient | $\ge 0.75$ |
| **Inference Latency** | Per-document runtime | $\le 1.5$ seconds |

---

## 📁 Repository Structure

```text
document-tampering-detection/
├── docs/                               # Project documentation & interim report
│   └── Document_Tampering_Interim_Report.pdf
├── notebooks/                          # Kaggle GPU training & experimental notebooks
├── src/                                # Modular Python package
│   ├── preprocessing/                  # ELA and Laplacian edge extractors
│   ├── data/                           # LMDB loader and batch generators
│   ├── models/                         # CNN classifier, U-Net, and Autoencoder
│   ├── explainability/                 # Grad-CAM heatmap generation
│   └── utils/                          # Metrics (Dice, IoU) and visualization
├── app/                                # Streamlit interactive web application
├── reports/                            # Final evaluation report & AI ethics analysis
├── requirements.txt                    # Project dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/titas-mahato/document-tampering-detection.git
cd document-tampering-detection
```

### 2. Set Up Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 👥 Team Details

* **Titas Mahato** — `SAU/CS/BTECH(CSE)/2024/69`
* **Tanmay Madan** — `SAU/CS/DMTECH(CSE)/2024/31`
* **Suyash Singh Gusain** — `SAU/CS/DMTECH(CSE)/2024/29`
* **Saara Ahmed** — `SAU/CS/BTECH(CSE)/2024/58`
* **Tarini Parashar** — `SAU/CS/BTECH(CSE)/2024/68`
* **Krish Singh** — `SAU/CS/BTECH(CSE)/2024/40`
* **Swarnojjwal Guha** — `SAU/CS/DMTECH(CSE)/2024/37`

**Department of Computer Science and Engineering**  
**Faculty of Engineering and Technology (FET)**  
**South Asian University (SAU)**
