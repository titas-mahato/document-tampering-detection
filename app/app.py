"""
DocTamper AI: Interactive Document Forgery & Tampering Detection Web Application
CSE310 - Neural Network and Deep Learning Semester-End Project
Faculty of Engineering & Technology, South Asian University
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="DocTamper AI | Document Forgery Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "app" / "samples"
WEIGHTS_DIR = BASE_DIR / "weights"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# --- FORENSIC PREPROCESSING (Imported from modular src package) ---
from src.preprocessing.ela import compute_ela, compute_ela_mean_energy
from src.preprocessing.edges import compute_laplacian_edges

# --- NEURAL NETWORK MODELS (Imported from modular src.models package) ---
try:
    import torch
    from src.models import DualStreamCNN, ResNet34UNet
    TORCH_AVAILABLE = True
except (ImportError, Exception):
    TORCH_AVAILABLE = False


@st.cache_resource
def load_pytorch_models(clf_path: str, unet_path: str):
    """Loads and caches PyTorch checkpoints for dual-stream classifier and U-Net localizer."""
    if not TORCH_AVAILABLE:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Classifier (CASIA Dual-Stream CNN)
    clf_model = DualStreamCNN(num_classes=2).to(device)
    c1 = torch.load(clf_path, map_location=device, weights_only=False)
    state1 = c1["model_state_dict"] if isinstance(c1, dict) and "model_state_dict" in c1 else c1
    clf_model.load_state_dict(state1)
    clf_model.eval()

    # 2. Load U-Net Localizer (DocTamper ResNet34 U-Net)
    unet_model = ResNet34UNet(pretrained=False).to(device)
    c2 = torch.load(unet_path, map_location=device, weights_only=False)
    state2 = c2["model_state_dict"] if isinstance(c2, dict) and "model_state_dict" in c2 else (c2["model"] if "model" in c2 else c2)
    unet_model.load_state_dict(state2)
    unet_model.eval()

    return clf_model, unet_model


def pytorch_inference(clf_model, unet_model, img_pil: Image.Image, ela_pil: Image.Image, threshold: float = 0.80):
    """Runs full two-stage deep learning inference using PyTorch models."""
    device = next(clf_model.parameters()).device
    orig_w, orig_h = img_pil.size

    # --- STAGE 1: CASIA DUAL-STREAM CLASSIFICATION (224x224) ---
    rgb_224 = img_pil.resize((224, 224))
    ela_224 = ela_pil.resize((224, 224))

    # Standard ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    rgb_arr = ((np.array(rgb_224, dtype=np.float32) / 255.0) - mean) / std
    ela_arr = ((np.array(ela_224, dtype=np.float32) / 255.0) - mean) / std

    rgb_t = torch.tensor(rgb_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    ela_t = torch.tensor(ela_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_clf = clf_model(rgb_t, ela_t)
        probs_clf = torch.softmax(logits_clf, dim=1).cpu().numpy()[0]

    # Class 0 = Authentic, Class 1 = Tampered
    tamper_prob = float(probs_clf[1])
    is_tampered = bool(tamper_prob > 0.5)
    confidence = tamper_prob if is_tampered else float(probs_clf[0])

    # --- STAGE 2: DOCTAMPER RESNET34 U-NET LOCALIZATION (512x512) ---
    rgb_512 = img_pil.resize((512, 512))
    rgb_512_arr = ((np.array(rgb_512, dtype=np.float32) / 255.0) - mean) / std
    seg_t = torch.tensor(rgb_512_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_logits = unet_model(seg_t)
        seg_probs = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]

    # Apply calibrated threshold (0.80 from DocTamper validation sweep)
    mask_binary = (seg_probs >= threshold).astype(np.uint8) * 255
    tamper_mask = cv2.resize(mask_binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    ela_np = np.array(ela_pil)
    ela_energy = compute_ela_mean_energy(ela_np)

    return is_tampered, confidence, tamper_mask, ela_energy, seg_probs


def heuristic_forensic_detection(img_pil: Image.Image, ela_pil: Image.Image, edges_np: np.ndarray):
    """Fallback heuristic analysis when PyTorch is not available in local environment."""
    ela_np = np.array(ela_pil)
    ela_energy = compute_ela_mean_energy(ela_np)
    gray_ela = np.mean(ela_np, axis=-1)
    ela_variance = float(np.var(gray_ela))

    anomaly_score = (ela_energy * 0.4) + (ela_variance * 0.05)
    threshold = np.percentile(gray_ela, 96.5)
    tamper_mask = (gray_ela > threshold).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    tamper_mask = cv2.dilate(tamper_mask, kernel, iterations=2)
    tamper_ratio = (np.sum(tamper_mask > 0) / tamper_mask.size) * 100.0

    is_tampered = bool(anomaly_score > 12.0 or tamper_ratio > 1.2)
    confidence = min(0.98, max(0.65, 0.50 + (abs(anomaly_score - 12.0) / 20.0)))
    return is_tampered, confidence, tamper_mask, ela_energy


def load_css():
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .status-authentic {
            color: #059669;
            font-weight: 700;
            font-size: 1.4rem;
        }
        .status-tampered {
            color: #DC2626;
            font-weight: 700;
            font-size: 1.4rem;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    load_css()
    st.markdown('<div class="main-header">🛡️ DocTamper AI: Document Forgery & Tampering Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">CSE310 Neural Network and Deep Learning | South Asian University</div>', unsafe_allow_html=True)

    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Control Panel")
        input_source = st.radio(
            "Select Document Source",
            ["Built-in Sample Documents", "Upload Custom Document"]
        )

        selected_image = None
        ground_truth_mask = None

        if input_source == "Built-in Sample Documents":
            sample_choice = st.selectbox(
                "Choose Sample",
                ["Authentic Invoice (Clean)", "Tampered Receipt (Forged Amount & Stamp)"]
            )
            if sample_choice == "Authentic Invoice (Clean)":
                img_path = SAMPLES_DIR / "authentic_invoice_01.png"
                if img_path.exists():
                    selected_image = Image.open(img_path)
            else:
                img_path = SAMPLES_DIR / "tampered_receipt_01.png"
                mask_path = SAMPLES_DIR / "tampered_receipt_01_mask.png"
                if img_path.exists():
                    selected_image = Image.open(img_path)
                if mask_path.exists():
                    ground_truth_mask = Image.open(mask_path)
        else:
            uploaded_file = st.file_uploader(
                "Upload Scanned Document (PNG / JPG / JPEG)",
                type=["png", "jpg", "jpeg"]
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)

        st.markdown("---")
        st.subheader("🔬 Forensic Parameters")
        ela_quality = st.slider("JPEG Compression Quality", min_value=60, max_value=98, value=90, step=2)
        ela_multiplier = st.slider("Error Enhancement Multiplier", min_value=5.0, max_value=40.0, value=25.0, step=5.0)
        seg_threshold = st.slider("U-Net Decision Threshold", min_value=0.10, max_value=0.95, value=0.80, step=0.05,
                                  help="Calibrated on DocTamper validation sweep. 0.80 minimizes false alarms.")

        st.markdown("---")
        st.caption("Developed by Titas Mahato & Team (SAU B.Tech/M.Tech CSE)")

    if selected_image is None:
        st.info("👈 Please select a sample document from the sidebar or upload an image to begin forensic inspection.")
        return

    # Ensure RGB
    selected_image = selected_image.convert("RGB")
    img_np = np.array(selected_image)

    # 1. Compute Forensic Visualizations
    with st.spinner("Executing forensic pipeline (ELA & Neural Inference)..."):
        ela_pil = compute_ela(selected_image, quality=ela_quality, multiplier=ela_multiplier)
        edges_np = compute_laplacian_edges(img_np)

        # Check for trained PyTorch weights
        clf_weights = WEIGHTS_DIR / "casia_dual_stream_best.pth"
        unet_weights = WEIGHTS_DIR / "doctamper_resnet34_unet_best.pth"

        neural_active = False
        if TORCH_AVAILABLE and clf_weights.exists() and unet_weights.exists():
            try:
                clf_model, unet_model = load_pytorch_models(str(clf_weights), str(unet_weights))
                if clf_model is not None and unet_model is not None:
                    is_tampered, confidence, tamper_mask, ela_energy, _ = pytorch_inference(
                        clf_model, unet_model, selected_image, ela_pil, threshold=seg_threshold
                    )
                    neural_active = True
                    st.success("🤖 Deep Learning Pipeline Active: Stage 1 (CASIA Dual-Stream) & Stage 2 (DocTamper ResNet34 U-Net) loaded from `weights/`.")
            except Exception as e:
                st.warning(f"PyTorch model execution encountered: {e}. Falling back to forensic heuristics.")

        if not neural_active:
            if not TORCH_AVAILABLE:
                st.info("ℹ️ Running in **Forensic Heuristic Mode** (Error Level Analysis + Laplacian Filtering). Running on Kaggle cloud will activate full neural inference.")
            else:
                st.info("ℹ️ Running in **Forensic Heuristic Mode**. Place weights in `weights/` to activate neural models.")
            is_tampered, confidence, tamper_mask, ela_energy = heuristic_forensic_detection(
                selected_image, ela_pil, edges_np
            )

    # 2. Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        verdict_class = "status-tampered" if is_tampered else "status-authentic"
        verdict_text = "⚠️ TAMPERED" if is_tampered else "✅ AUTHENTIC"
        st.markdown(f"**Document Verdict**<br><span class='{verdict_class}'>{verdict_text}</span>", unsafe_allow_html=True)
    with col2:
        st.metric("Model Confidence", f"{confidence * 100:.1f} %")
    with col3:
        st.metric("Mean ELA Residual", f"{ela_energy:.2f}")
    with col4:
        altered_pct = (np.sum(tamper_mask > 0) / tamper_mask.size) * 100.0 if is_tampered else 0.0
        st.metric("Altered Area Ratio", f"{altered_pct:.2f} %")

    st.markdown("---")

    # 3. Four-Panel Inspection View
    st.subheader("🔍 Multi-Stream Forensic Breakdown")
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.markdown("##### 1. Original Document")
        st.image(selected_image, use_container_width=True)

    with p2:
        st.markdown("##### 2. ELA Compression Residual")
        st.image(ela_pil, use_container_width=True, caption=f"Quality: {ela_quality}% | Boost: {ela_multiplier}x")

    with p3:
        st.markdown("##### 3. Laplacian Micro-Edges")
        st.image(edges_np, use_container_width=True, caption="High-pass splicing edge anomalies")

    with p4:
        st.markdown("##### 4. Tampering Localization Mask")
        overlay = img_np.copy()
        if is_tampered:
            overlay[tamper_mask > 127] = [230, 40, 40]
            st.image(overlay, use_container_width=True, caption="Red: Detected Forged/Spliced Regions")
        else:
            st.image(selected_image, use_container_width=True, caption="No significant tamper regions detected")

    if ground_truth_mask is not None:
        st.markdown("---")
        st.subheader("🎯 Ground Truth Mask Comparison (DocTamper Benchmark)")
        g1, g2 = st.columns(2)
        with g1:
            st.image(ground_truth_mask, caption="Official Benchmark Binary Label Mask", width=350)
        with g2:
            st.image(tamper_mask, caption="Predicted Tampering Segmentation Mask", width=350)


if __name__ == "__main__":
    main()
