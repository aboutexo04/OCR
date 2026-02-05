"""
Ensemble Inference for High Precision OCR
Combines multiple models for better accuracy

Usage:
    python ensemble_inference.py
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
BASE_PATH = './data/datasets'
TEST_IMG_DIR = os.path.join(BASE_PATH, 'images/test')
TEST_JSON = os.path.join(BASE_PATH, 'jsons/test.json')
SAMPLE_SUB = os.path.join(BASE_PATH, 'sample_submission.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== Model Definitions ==========
class HighPrecisionOCRModel(nn.Module):
    def __init__(self, encoder_name, encoder_weights=None):
        super().__init__()
        self.base_model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            decoder_attention_type='scse'
        )
        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        return self.refine(self.base_model(x))


class SimpleOCRModel(nn.Module):
    """Simpler model for diversity in ensemble"""
    def __init__(self, encoder_name, encoder_weights=None):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1
        )

    def forward(self, x):
        return self.model(x)

# ========== Post-processing ==========
def apply_morphology(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def refine_mask_with_watershed(mask):
    """Use watershed for better polygon separation"""
    mask_uint8 = (mask * 255).astype(np.uint8)

    # Find sure background
    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask_uint8, kernel, iterations=2)

    # Find sure foreground
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.3 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Apply watershed
    mask_color = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(mask_color, markers)

    # Create refined mask
    refined = np.zeros_like(mask_uint8)
    refined[markers > 1] = 255

    return refined


def mask_to_polygons_advanced(mask, min_area=50, epsilon_factor=0.003, use_watershed=True):
    """Convert mask to polygons with advanced filtering"""
    # Apply morphology
    mask_processed = apply_morphology((mask * 255).astype(np.uint8))

    # Optional watershed refinement
    if use_watershed:
        try:
            mask_processed = refine_mask_with_watershed(mask_processed / 255.0)
        except:
            pass

    # Find contours
    contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        # Bounding rect check
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 3:
            continue

        # Aspect ratio filter (typical text has aspect ratio between 0.1 and 30)
        aspect = w / (h + 1e-8)
        if aspect < 0.1 or aspect > 50:
            continue

        # Simplify contour
        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) >= 4:
            # Use minimum area rectangle for very irregular shapes
            if len(approx) > 10:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                approx = np.int0(box).reshape(-1, 1, 2)

            polygons.append(approx.reshape(-1, 2).tolist())

    return polygons


def polygons_to_string(polygons):
    if not polygons:
        return ""
    return "|".join([" ".join([f"{int(p[0])} {int(p[1])}" for p in poly]) for poly in polygons])

# ========== Transform ==========
def get_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ========== Multi-scale TTA ==========
def predict_single_model(model, img_tensor, device, scales=[1.0, 0.75, 1.25]):
    """Predict with single model using multi-scale TTA"""
    model.eval()
    _, h, w = img_tensor.shape
    all_preds = []

    with torch.no_grad():
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            new_h = max((new_h // 32) * 32, 32)
            new_w = max((new_w // 32) * 32, 32)

            scaled = F.interpolate(img_tensor.unsqueeze(0), size=(new_h, new_w),
                                   mode='bilinear', align_corners=False).to(device)

            with autocast('cuda', dtype=torch.bfloat16):
                pred = torch.sigmoid(model(scaled))
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
            all_preds.append(pred.float())

            # Horizontal flip
            with autocast('cuda', dtype=torch.bfloat16):
                pred_flip = torch.sigmoid(model(torch.flip(scaled, dims=[3])))
            pred_flip = torch.flip(pred_flip, dims=[3])
            pred_flip = F.interpolate(pred_flip, size=(h, w), mode='bilinear', align_corners=False)
            all_preds.append(pred_flip.float())

    return torch.stack(all_preds, dim=0).mean(dim=0)


def ensemble_predict(models, img_tensor, device, scales=[1.0, 0.75, 1.25], weights=None):
    """Ensemble prediction from multiple models"""
    if weights is None:
        weights = [1.0] * len(models)

    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    all_preds = []
    for model, weight in zip(models, weights):
        pred = predict_single_model(model, img_tensor, device, scales)
        all_preds.append(pred * weight)

    ensemble_pred = sum(all_preds)
    return ensemble_pred

# ========== Main Ensemble Inference ==========
def run_ensemble_inference():
    print("=" * 60)
    print("ENSEMBLE INFERENCE FOR HIGH PRECISION OCR")
    print("=" * 60)

    # Define model configurations
    model_configs = [
        # (model_class, encoder_name, weight_path, ensemble_weight)
        ('high', 'tu-convnext_base', 'final_model.pth', 1.0),
        ('high', 'tu-convnext_base', 'base_model.pth', 0.8),
        ('simple', 'efficientnet-b3', 'best_model.pth', 0.6),
    ]

    # Load available models
    models = []
    weights = []

    for model_type, encoder, weight_path, ensemble_weight in model_configs:
        if not os.path.exists(weight_path):
            print(f"Skip: {weight_path} not found")
            continue

        print(f"Loading: {weight_path} ({encoder})")
        try:
            if model_type == 'high':
                model = HighPrecisionOCRModel(encoder, None).to(DEVICE)
            else:
                model = SimpleOCRModel(encoder, None).to(DEVICE)

            model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            model.eval()
            models.append(model)
            weights.append(ensemble_weight)
            print(f"  Loaded successfully")
        except Exception as e:
            print(f"  Failed: {e}")

    if not models:
        print("\nNo models found! Please train models first.")
        print("Run: python train_with_pseudo_labels.py --stage 1")
        return

    print(f"\nEnsemble: {len(models)} models")

    # Prepare test data
    transform = get_transform(1536)

    with open(TEST_JSON, 'r', encoding='utf-8') as f:
        test_data = json.load(f)['images']

    # Inference settings
    threshold = 0.5
    min_area = 50
    scales = [1.0, 0.75, 1.25]

    preds = {}

    for img_name in tqdm(test_data.keys(), desc="Ensemble Inference"):
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            preds[img_name] = ""
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # Transform
        transformed = transform(image=image)
        img_tensor = transformed['image']

        # Ensemble prediction
        pred = ensemble_predict(models, img_tensor, DEVICE, scales, weights)

        # Resize to original
        pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        mask = pred.cpu().numpy()[0, 0]

        # Post-processing
        polygons = mask_to_polygons_advanced(mask > threshold, min_area=min_area,
                                             use_watershed=True)
        preds[img_name] = polygons_to_string(polygons)

    # Save submission
    sample_df = pd.read_csv(SAMPLE_SUB)
    sample_df['polygons'] = sample_df['filename'].map(preds).fillna("")

    output_csv = 'submission_ensemble.csv'
    sample_df.to_csv(output_csv, index=False)

    # Statistics
    counts = sample_df['polygons'].apply(lambda x: len(x.split('|')) if x else 0)
    print(f"\n{'=' * 60}")
    print(f"Saved: {output_csv}")
    print(f"Total images: {len(sample_df)}")
    print(f"Avg polygons/image: {counts.mean():.1f}")
    print(f"Max: {counts.max()}, Min: {counts.min()}")
    print(f"{'=' * 60}")


def run_threshold_sweep():
    """Find optimal threshold by analyzing predictions"""
    print("\n" + "=" * 60)
    print("THRESHOLD SWEEP ANALYSIS")
    print("=" * 60)

    # This would typically be done on validation data
    # For now, just show recommended thresholds

    print("""
Recommended Thresholds based on typical OCR characteristics:

    For High Precision (fewer false positives):
    - threshold = 0.6 to 0.7
    - min_area = 80 to 100

    For High Recall (fewer false negatives):
    - threshold = 0.3 to 0.4
    - min_area = 30 to 50

    Balanced:
    - threshold = 0.5
    - min_area = 50

Tips:
1. If you're getting too many false detections, increase threshold
2. If you're missing small text, decrease min_area
3. Use watershed=True for better polygon separation
""")


if __name__ == "__main__":
    run_ensemble_inference()
    run_threshold_sweep()
