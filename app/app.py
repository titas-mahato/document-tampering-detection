"""
DocTamper AI: Enterprise Document Forgery Detection & Localization Platform
CSE310 - Neural Network and Deep Learning Semester-End Project
Faculty of Engineering & Technology, South Asian University

Two-Stage Pipeline Architecture:
- Stage 1: CASIA Dual-Stream CNN (RGB + ELA) for Document Authenticity Screening
- Stage 2: DocTamper ResNet34 U-Net with Skip Connections for Pixel-Level Tampering Localization
"""

import os
import sys
import io
import json
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import matplotlib.pyplot as plt
import streamlit as st

# Configure wide layout
st.set_page_config(
    page_title="DocTamper AI | Forensic Document Verification",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "app" / "samples"
WEIGHTS_DIR = BASE_DIR / "weights"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Fixed Training Hyperparameters (Grounded in Training Notebooks)
CASIA_IMG_SIZE = (224, 224)
CASIA_ELA_QUALITY = 90
DOCTAMPER_IMG_SIZE = (512, 512)
DEFAULT_THRESHOLD = 0.80  # Calibrated on DocTamper validation sweep


# =====================================================================
# 1. DOMAIN VALIDATION (OOD / Document-Type Verification)
# =====================================================================
def validate_document_domain(img_pil: Image.Image) -> Tuple[bool, str, Dict[str, float]]:
    """
    Evaluates whether an image falls within the valid distribution of scanned
    financial, administrative, or legal documents (vs. cartoons, illustrations, natural photos).
    """
    img_rgb = np.array(img_pil.convert("RGB"))
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Metric 1: Mean Color Saturation (Documents are largely monochromatic; cartoons/photos have high saturation)
    mean_saturation = float(np.mean(hsv[:, :, 1])) / 255.0

    # Metric 2: Background Luminance (Paper documents typically have light/white paper backgrounds)
    mean_luminance = float(np.mean(hsv[:, :, 2])) / 255.0

    # Metric 3: High-contrast stroke density (Documents contain sharp text edges against flat backgrounds)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.mean(edges > 0))

    metrics = {
        "mean_saturation": mean_saturation,
        "mean_luminance": mean_luminance,
        "edge_density": edge_density
    }

    # Decision heuristics:
    # High saturation (>0.32) indicates colorful illustrations, cartoons, or vibrant artwork.
    # Very dark luminance (<0.35) indicates dark-mode screenshots or natural night scenes.
    if mean_saturation > 0.32:
        return False, "High color saturation detected (characteristic of cartoons, illustrations, or artwork rather than paper documents).", metrics
    if mean_luminance < 0.30:
        return False, "Low background luminance detected (paper documents typically feature light/white backgrounds).", metrics

    return True, "Valid document domain (scanned paper / form characteristics verified).", metrics


# =====================================================================
# 2. EXACT ELA PIPELINE (From doctamper-ai-casia-classification.ipynb Cell 26)
# =====================================================================
def generate_ela_image(image: Image.Image, quality: int = CASIA_ELA_QUALITY) -> Image.Image:
    """
    Computes Error Level Analysis using an in-memory BytesIO buffer.
    Exact implementation from CASIA training notebook.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    ela = ImageChops.difference(image, recompressed)
    extrema = ela.getextrema()
    max_difference = max(channel[1] for channel in extrema)

    if max_difference > 0:
        ela = ImageEnhance.Brightness(ela).enhance(255.0 / max_difference)

    return ela


# =====================================================================
# 3. PYTORCH ARCHITECTURE DEFINITIONS
# =====================================================================
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class CNNFeatureExtractor(nn.Module):
        """Feature extractor branch from doctamper-ai-casia-classification.ipynb (Cell 30)."""
        def __init__(self, in_channels: int = 3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.flatten(self.features(x), 1)

    class DualStreamCNN(nn.Module):
        """Dual-Stream Classifier (RGB + ELA) from doctamper-ai-casia-classification.ipynb."""
        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.rgb_branch = CNNFeatureExtractor(in_channels=3)
            self.ela_branch = CNNFeatureExtractor(in_channels=3)
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )

        def forward(self, rgb: torch.Tensor, ela: torch.Tensor) -> torch.Tensor:
            rgb_f = self.rgb_branch(rgb)
            ela_f = self.ela_branch(ela)
            fused = torch.cat((rgb_f, ela_f), dim=1)
            return self.classifier(fused)

    class DoubleConv(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    class DecoderBlock(nn.Module):
        def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
            super().__init__()
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)

        def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return self.conv(torch.cat([x, skip], dim=1))

    class ResNet34UNet(nn.Module):
        """ResNet34 U-Net architecture from DocTamperAI_Clean_Segmentation.ipynb (Cell 16)."""
        def __init__(self):
            super().__init__()
            backbone = models.resnet34(weights=None)
            self.encoder0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
            self.pool = backbone.maxpool
            self.encoder1 = backbone.layer1
            self.encoder2 = backbone.layer2
            self.encoder3 = backbone.layer3
            self.encoder4 = backbone.layer4

            self.decoder4 = DecoderBlock(512, 256, 256)
            self.decoder3 = DecoderBlock(256, 128, 128)
            self.decoder2 = DecoderBlock(128, 64, 64)
            self.decoder1 = DecoderBlock(64, 64, 64)

            self.final_up = nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.output = nn.Conv2d(32, 1, kernel_size=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            orig_size = x.shape[-2:]
            e0 = self.encoder0(x)
            e1 = self.encoder1(self.pool(e0))
            e2 = self.encoder2(e1)
            e3 = self.encoder3(e2)
            e4 = self.encoder4(e3)

            d4 = self.decoder4(e4, e3)
            d3 = self.decoder3(d4, e2)
            d2 = self.decoder2(d3, e1)
            d1 = self.decoder1(d2, e0)

            x = nn.functional.interpolate(d1, size=orig_size, mode="bilinear", align_corners=False)
            return self.output(self.final_up(x))


@st.cache_resource
def load_pytorch_models(clf_path: str, unet_path: str):
    """Loads and caches both trained PyTorch checkpoints."""
    if not TORCH_AVAILABLE:
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load CASIA Classifier
    clf_model = DualStreamCNN(num_classes=2).to(device)
    c1 = torch.load(clf_path, map_location=device, weights_only=False)
    state1 = c1["model_state_dict"] if isinstance(c1, dict) and "model_state_dict" in c1 else c1
    clf_model.load_state_dict(state1)
    clf_model.eval()

    # 2. Load DocTamper U-Net Localizer
    unet_model = ResNet34UNet().to(device)
    c2 = torch.load(unet_path, map_location=device, weights_only=False)
    state2 = c2["model_state_dict"] if isinstance(c2, dict) and "model_state_dict" in c2 else (c2["model"] if "model" in c2 else c2)
    unet_model.load_state_dict(state2)
    unet_model.eval()

    return clf_model, unet_model


def run_pytorch_inference(clf_model, unet_model, img_pil: Image.Image, threshold: float = DEFAULT_THRESHOLD):
    """
    Executes true deep learning inference matching both training notebooks:
    - CASIA: inputs in [0, 1] range (transforms.ToTensor(), NO ImageNet standardization).
    - DocTamper: inputs in ImageNet standardized range.
    """
    device = next(clf_model.parameters()).device
    orig_w, orig_h = img_pil.size

    # --- 1. Compute Fixed-Parameter ELA ---
    ela_pil = generate_ela_image(img_pil, quality=CASIA_ELA_QUALITY)

    # --- 2. Stage 1: CASIA Dual-Stream Inference (224x224) ---
    rgb_224 = img_pil.resize(CASIA_IMG_SIZE)
    ela_224 = ela_pil.resize(CASIA_IMG_SIZE)

    rgb_arr = np.array(rgb_224, dtype=np.float32) / 255.0
    ela_arr = np.array(ela_224, dtype=np.float32) / 255.0

    rgb_t = torch.tensor(rgb_arr).permute(2, 0, 1).unsqueeze(0).to(device)
    ela_t = torch.tensor(ela_arr).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits_clf = clf_model(rgb_t, ela_t)
        probs_clf = torch.softmax(logits_clf, dim=1).cpu().numpy()[0]

    # Class 0: Authentic, Class 1: Tampered
    tamper_prob = float(probs_clf[1])
    is_tampered = bool(tamper_prob >= 0.50)
    confidence = tamper_prob if is_tampered else float(probs_clf[0])

    # --- 3. Stage 2: DocTamper ResNet34 U-Net Inference (512x512) ---
    rgb_512 = img_pil.resize(DOCTAMPER_IMG_SIZE)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb_512_norm = ((np.array(rgb_512, dtype=np.float32) / 255.0) - mean) / std

    seg_t = torch.tensor(rgb_512_norm).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_logits = unet_model(seg_t)
        # Apply sigmoid ONCE to raw model logits
        seg_probs_512 = torch.sigmoid(seg_logits.float()).cpu().numpy()[0, 0]

    # Resize continuous probability map back to original image dimensions
    seg_probs = cv2.resize(seg_probs_512, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    peak_prob = float(np.max(seg_probs))

    # Binary mask via thresholding
    mask_binary = (seg_probs >= threshold).astype(np.uint8) * 255
    altered_ratio = (np.sum(mask_binary > 0) / mask_binary.size) * 100.0

    return is_tampered, confidence, seg_probs, mask_binary, peak_prob, altered_ratio, ela_pil


def run_fallback_inference(img_pil: Image.Image, threshold: float = DEFAULT_THRESHOLD):
    """
    Lightweight heuristic fallback when PyTorch is not available locally.
    Does not use flawed percentile thresholds.
    """
    orig_w, orig_h = img_pil.size
    ela_pil = generate_ela_image(img_pil, quality=CASIA_ELA_QUALITY)
    ela_np = np.array(ela_pil)
    gray_ela = np.mean(ela_np, axis=-1)
    ela_std = float(np.std(gray_ela))

    # Normalized gradient map
    norm_ela = np.clip((gray_ela - 12.0) / 60.0, 0.0, 1.0)
    seg_probs = cv2.resize(norm_ela, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    peak_prob = float(np.max(seg_probs))

    mask_binary = (seg_probs >= threshold).astype(np.uint8) * 255
    altered_ratio = (np.sum(mask_binary > 0) / mask_binary.size) * 100.0

    is_tampered = bool(ela_std > 22.0 or altered_ratio > 0.5)
    confidence = min(0.95, max(0.60, (ela_std / 35.0) if is_tampered else (1.0 - ela_std / 35.0)))

    return is_tampered, confidence, seg_probs, mask_binary, peak_prob, altered_ratio, ela_pil


# =====================================================================
# 4. EVALUATION & METRICS HELPERS
# =====================================================================
def compute_dice_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Tuple[float, float]:
    """Computes exact Dice coefficient and Intersection over Union between two binary masks."""
    p = (pred_mask > 0).astype(bool)
    g = (gt_mask > 0).astype(bool)

    intersection = np.logical_and(p, g).sum()
    total_p = p.sum()
    total_g = g.sum()
    union = np.logical_or(p, g).sum()

    if total_p + total_g == 0:
        return 1.0, 1.0
    dice = (2.0 * intersection) / (total_p + total_g + 1e-7)
    iou = intersection / (union + 1e-7) if union > 0 else 0.0
    return float(dice), float(iou)


def render_heatmap(seg_probs: np.ndarray) -> np.ndarray:
    """Converts a [0.0, 1.0] probability map into a continuous RGB heatmap."""
    probs_uint8 = np.uint8(np.clip(seg_probs * 255.0, 0, 255))
    heatmap_bgr = cv2.applyColorMap(probs_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def render_overlay(orig_rgb: np.ndarray, heatmap_rgb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blends the original document with the continuous heatmap using adjustable opacity."""
    return cv2.addWeighted(heatmap_rgb, alpha, orig_rgb, 1.0 - alpha, 0)


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


def img_to_bytes(img_np: np.ndarray) -> bytes:
    pil_img = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# =====================================================================
# 5. UI STYLING & DASHBOARD
# =====================================================================
def load_custom_css():
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.1rem;
            font-weight: 800;
            color: #1E3A8A;
            margin-bottom: 0.1rem;
        }
        .sub-header {
            font-size: 1.0rem;
            color: #4B5563;
            margin-bottom: 1.2rem;
        }
        .metric-box {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .verdict-auth {
            color: #059669;
            font-weight: 800;
            font-size: 1.35rem;
        }
        .verdict-tamp {
            color: #DC2626;
            font-weight: 800;
            font-size: 1.35rem;
        }
        </style>
    """, unsafe_allow_html=True)


def main():
    load_custom_css()
    st.markdown('<div class="main-header">🛡️ DocTamper AI: Document Forgery Detection & Localization</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">CSE310 Neural Network and Deep Learning | South Asian University</div>', unsafe_allow_html=True)

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        input_source = st.radio(
            "Document Source",
            ["Built-in Benchmark Samples", "Upload Custom Document"]
        )

        selected_image = None
        ground_truth_mask = None
        sample_title = ""

        if input_source == "Built-in Benchmark Samples":
            sample_choice = st.selectbox(
                "Select Verified Sample",
                ["Authentic Invoice (Clean Document)", "Tampered Receipt (Forged Total & Stamp)"]
            )
            sample_title = sample_choice
            if sample_choice == "Authentic Invoice (Clean Document)":
                img_path = SAMPLES_DIR / "authentic_invoice_01.png"
                if img_path.exists():
                    selected_image = Image.open(img_path)
            else:
                img_path = SAMPLES_DIR / "tampered_receipt_01.png"
                mask_path = SAMPLES_DIR / "tampered_receipt_01_mask.png"
                if img_path.exists():
                    selected_image = Image.open(img_path)
                if mask_path.exists():
                    ground_truth_mask = np.array(Image.open(mask_path).convert("L"))
        else:
            uploaded_file = st.file_uploader(
                "Upload Image (PNG, JPG, JPEG, TIFF)",
                type=["png", "jpg", "jpeg", "tif", "tiff"]
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                sample_title = uploaded_file.name

        st.markdown("---")
        st.subheader("🔬 Localization Controls")
        threshold_slider = st.slider(
            "Decision Threshold (τ)",
            min_value=0.10, max_value=0.95, value=DEFAULT_THRESHOLD, step=0.05,
            help="Optimal F1 threshold calibrated on DocTamper validation sweep is 0.80. Lower values increase recall; higher values reduce false alarms."
        )
        overlay_alpha = st.slider(
            "Heatmap Overlay Opacity",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="Adjust alpha transparency of the probability heatmap overlay over the original document."
        )

        st.markdown("---")
        st.caption("DocTamper AI Team: Titas Mahato, Tanmay & Team")

    if selected_image is None:
        st.info("👈 Please select a benchmark sample document or upload an image from the sidebar to begin inspection.")
        return

    # Ensure clean RGB format
    selected_image = selected_image.convert("RGB")
    orig_np = np.array(selected_image)

    # --- Domain Validation ---
    is_doc, domain_msg, domain_metrics = validate_document_domain(selected_image)
    if not is_doc:
        st.warning(f"⚠️ **Out-of-Domain Notice:** {domain_msg}\n\n*DocTamper AI was trained specifically on scanned invoices, receipts, tax forms, and administrative documents. Processing cartoons, artwork, or non-document photographs can cause domain shift, resulting in overconfident false alarms.*")

    # --- Inference Execution ---
    with st.spinner("Executing Two-Stage Neural Inspection Pipeline..."):
        clf_weights = WEIGHTS_DIR / "casia_dual_stream_best.pth"
        unet_weights = WEIGHTS_DIR / "doctamper_resnet34_unet_best.pth"

        neural_active = False
        if TORCH_AVAILABLE and clf_weights.exists() and unet_weights.exists():
            try:
                clf_model, unet_model = load_pytorch_models(str(clf_weights), str(unet_weights))
                if clf_model is not None and unet_model is not None:
                    is_tampered, confidence, seg_probs, mask_binary, peak_prob, altered_ratio, ela_pil = run_pytorch_inference(
                        clf_model, unet_model, selected_image, threshold=threshold_slider
                    )
                    neural_active = True
            except Exception as e:
                st.error(f"Neural inference exception: {e}. Executing fallback mode.")

        if not neural_active:
            is_tampered, confidence, seg_probs, mask_binary, peak_prob, altered_ratio, ela_pil = run_fallback_inference(
                selected_image, threshold=threshold_slider
            )

    # Render visualizations
    heatmap_rgb = render_heatmap(seg_probs)
    overlay_rgb = render_overlay(orig_np, heatmap_rgb, alpha=overlay_alpha)

    # --- Section 1: Executive KPI Metrics ---
    st.markdown("### 📊 Forensic Assessment Overview")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        v_class = "verdict-tamp" if is_tampered else "verdict-auth"
        v_text = "⚠️ TAMPERED" if is_tampered else "✅ AUTHENTIC"
        st.markdown(f"**Stage 1 Verdict**<br><span class='{v_class}'>{v_text}</span>", unsafe_allow_html=True)
    with kpi2:
        st.metric("CASIA Confidence", f"{confidence * 100:.1f} %", help="Probability score output by the CASIA Dual-Stream CNN")
    with kpi3:
        st.metric("Peak Tamper Probability", f"{peak_prob * 100:.1f} %", help="Maximum pixel probability produced by U-Net")
    with kpi4:
        st.metric("Altered Area Ratio", f"{altered_ratio:.2f} %", help=f"Percentage of total document pixels exceeding threshold {threshold_slider:.2f}")
    with kpi5:
        mode_badge = "🟢 PyTorch Neural" if neural_active else "🟡 Fallback"
        st.metric("Inference Engine", mode_badge)

    st.markdown("---")

    # --- Section 2: Four-Stage Visual Inspection Layout ---
    st.markdown("### 🔍 Four-Stage Localization Inspection View")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("##### 1. Original Document")
        st.image(selected_image, use_container_width=True, caption=f"Resolution: {orig_np.shape[1]}x{orig_np.shape[0]}")

    with col_b:
        st.markdown(f"##### 2. Binary Tamper Mask (τ={threshold_slider:.2f})")
        st.image(mask_binary, use_container_width=True, caption=f"Altered Pixels: {int(np.sum(mask_binary > 0)):,}")

    with col_c:
        st.markdown("##### 3. Probability Heatmap")
        st.image(heatmap_rgb, use_container_width=True, caption="Jet Colormap: Blue (0.0) → Red (1.0)")

    with col_d:
        st.markdown(f"##### 4. Heatmap Overlay (α={overlay_alpha:.2f})")
        st.image(overlay_rgb, use_container_width=True, caption="Aligned Semi-Transparent Fusion")

    # --- Section 3: Ground-Truth Mask Evaluation (When Sample Mask is Present) ---
    if ground_truth_mask is not None:
        st.markdown("---")
        st.markdown("### 🎯 Ground-Truth Benchmark Localization Validation")
        gt_col1, gt_col2, gt_col3 = st.columns([1, 1, 1])

        # Ensure ground truth matches dimensions
        if ground_truth_mask.shape != mask_binary.shape:
            ground_truth_mask = cv2.resize(ground_truth_mask, (mask_binary.shape[1], mask_binary.shape[0]), interpolation=cv2.INTER_NEAREST)

        sample_dice, sample_iou = compute_dice_iou(mask_binary, ground_truth_mask)

        with gt_col1:
            st.image(ground_truth_mask, caption="Official Benchmark Ground-Truth Label", use_container_width=True)
        with gt_col2:
            st.image(mask_binary, caption=f"Model Predicted Mask (τ = {threshold_slider:.2f})", use_container_width=True)
        with gt_col3:
            st.markdown("#### Real-Time Mask Overlap Metrics")
            st.metric("Sample Dice Coefficient", f"{sample_dice * 100:.2f} %")
            st.metric("Sample IoU (Jaccard Index)", f"{sample_iou * 100:.2f} %")
            st.caption("Dice & IoU update dynamically as you adjust the threshold slider in the sidebar.")

    st.markdown("---")

    # --- Section 4: Detailed Probability Distribution & Benchmark Context ---
    tab_dist, tab_benchmarks, tab_export = st.tabs([
        "📈 Pixel Probability Distribution",
        "🏆 Model Benchmark Performance",
        "📥 Export Reports & Mask Assets"
    ])

    with tab_dist:
        st.markdown("#### Spatial Pixel Probability Spectrum")
        st.caption("Visualizes the separation between authentic document background pixels and high-confidence tampered pixels.")

        fig, ax = plt.subplots(figsize=(10, 3.2))
        probs_flat = seg_probs.flatten()
        # Sample for plotting speed if image is huge
        if len(probs_flat) > 200000:
            probs_flat = np.random.choice(probs_flat, size=200000, replace=False)

        ax.hist(probs_flat, bins=50, range=(0.0, 1.0), color="#2563EB", alpha=0.75, edgecolor="black")
        ax.axvline(x=threshold_slider, color="#DC2626", linestyle="--", linewidth=2, label=f"Current Threshold (τ = {threshold_slider:.2f})")
        ax.set_yscale("log")
        ax.set_xlabel("Predicted Tamper Probability", fontsize=11)
        ax.set_ylabel("Pixel Count (Log Scale)", fontsize=11)
        ax.set_title("U-Net Output Probability Histogram", fontsize=12, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with tab_benchmarks:
        st.markdown("#### Verified Experimental Test Results")
        st.caption("Grounded in official test evaluations from team training checkpoints (no synthetic metrics).")

        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.markdown("##### Stage 1: CASIA Dual-Stream CNN")
            st.table({
                "Evaluation Metric": ["Test Accuracy", "Test Precision", "Test Recall", "Test F1-Score", "ROC-AUC Score", "Evaluated Samples"],
                "Achieved Value": ["90.32 %", "83.16 %", "98.57 %", "90.21 %", "97.52 %", "2,004 Images (CASIA Test Split)"]
            })
            st.caption("Confusion Matrix on Test Set: TP=894, TN=916, FP=181, FN=13.")

        with m_col2:
            st.markdown("##### Stage 2: DocTamper ResNet34 U-Net (V1 Baseline)")
            st.table({
                "Evaluation Metric": ["Held-Out Test Dice", "Held-Out Test IoU", "Pixel-Level Precision", "Pixel-Level Recall", "Calibrated Threshold", "Evaluated Samples"],
                "Achieved Value": ["34.18 %", "26.10 %", "20.18 %", "68.81 %", "0.80", "30,000 Documents (DocTamper TestingSet)"]
            })
            st.caption("High recall (68.81%) confirms effective localization of subtle character edits.")

    with tab_export:
        st.markdown("#### Export Forensic Analysis Package")
        exp1, exp2, exp3, exp4 = st.columns(4)

        with exp1:
            st.download_button(
                "⬇️ Download Binary Mask",
                data=img_to_bytes(mask_binary),
                file_name="doctamper_binary_mask.png",
                mime="image/png"
            )
        with exp2:
            st.download_button(
                "⬇️ Download Heatmap",
                data=img_to_bytes(heatmap_rgb),
                file_name="doctamper_probability_heatmap.png",
                mime="image/png"
            )
        with exp3:
            st.download_button(
                "⬇️ Download Overlay",
                data=img_to_bytes(overlay_rgb),
                file_name="doctamper_heatmap_overlay.png",
                mime="image/png"
            )
        with exp4:
            # Build JSON Metadata
            report_data = {
                "project": "DocTamper AI",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sample_analyzed": sample_title,
                "is_document_domain": is_doc,
                "domain_diagnostics": domain_metrics,
                "stage_1_classification": {
                    "verdict": "TAMPERED" if is_tampered else "AUTHENTIC",
                    "confidence": confidence,
                    "model": "CASIA Dual-Stream CNN"
                },
                "stage_2_localization": {
                    "threshold_applied": threshold_slider,
                    "altered_area_percentage": altered_ratio,
                    "peak_pixel_probability": peak_prob,
                    "model": "DocTamper ResNet34 U-Net"
                }
            }
            st.download_button(
                "⬇️ Export Forensic JSON",
                data=json.dumps(report_data, indent=2),
                file_name="doctamper_forensic_report.json",
                mime="application/json"
            )


if __name__ == "__main__":
    main()
