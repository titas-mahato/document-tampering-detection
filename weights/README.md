# Model Weights Directory

This folder holds trained deep learning model weights exported from Kaggle GPU runs.

## Expected Weight Files
When training completes on Kaggle, download the saved weights into this folder:

| Filename | Architecture | Task | Input Size |
| :--- | :--- | :--- | :--- |
| `classifier_best.h5` | Dual-Stream CNN (MobileNetV2 + ELA) | Document Classification | `(512, 512, 3)` x 2 |
| `unet_best.h5` | U-Net with Skip Connections | Tampered Region Localization | `(512, 512, 3)` |
| `autoencoder_best.h5` | Convolutional Denoising Autoencoder | Anomaly Reconstruction | `(256, 256, 3)` |

## Note on Git Tracking
Model weight files (`*.h5`, `*.keras`, `*.weights`) are excluded by `.gitignore` to prevent bloating the GitHub repository.

## Live Demo Fallback
If the Streamlit web app (`app/app.py`) is run without these weights present, it gracefully activates a **Forensic Heuristic Mode** using live Error Level Analysis and Laplacian edge detection, ensuring your demonstration never crashes during presentations.
