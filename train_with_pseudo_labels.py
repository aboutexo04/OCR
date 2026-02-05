"""
High Precision OCR Training with Pseudo Labels
Target: 95+ Score

Usage:
    python train_with_pseudo_labels.py --stage 1  # Train base model
    python train_with_pseudo_labels.py --stage 2  # Generate pseudo labels
    python train_with_pseudo_labels.py --stage 3  # Train with pseudo labels
    python train_with_pseudo_labels.py --stage 4  # Final inference
"""

import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.amp import autocast, GradScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm
import gc
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ========== Configuration ==========
class Config:
    # Paths
    BASE_PATH = './data/datasets'
    TRAIN_IMG_DIR = os.path.join(BASE_PATH, 'images/train')
    VAL_IMG_DIR = os.path.join(BASE_PATH, 'images/val')
    TEST_IMG_DIR = os.path.join(BASE_PATH, 'images/test')
    TRAIN_JSON = os.path.join(BASE_PATH, 'jsons/train.json')
    VAL_JSON = os.path.join(BASE_PATH, 'jsons/val.json')
    TEST_JSON = os.path.join(BASE_PATH, 'jsons/test.json')
    SAMPLE_SUB = os.path.join(BASE_PATH, 'sample_submission.csv')

    # Pseudo Label Paths
    PSEUDO_BASE = './data/pseudo_label'
    PSEUDO_DIRS = [
        os.path.join(PSEUDO_BASE, 'sroie/images/train'),
        os.path.join(PSEUDO_BASE, 'sroie/images/test'),
        os.path.join(PSEUDO_BASE, 'wildreceipt/images'),
        os.path.join(PSEUDO_BASE, 'cord-v2/images'),
    ]
    PSEUDO_MASK_DIR = './pseudo_masks'

    # Model
    ENCODER = 'tu-convnext_base'
    ENCODER_WEIGHTS = 'imagenet'

    # Training Stage 1 (Base)
    RESIZE_TARGET = 1536
    BATCH_SIZE = 2
    ACCUMULATION_STEPS = 16
    EPOCHS_STAGE1 = 15   # Reduced from 30
    LEARNING_RATE = 1e-4

    # Training Stage 3 (With Pseudo)
    EPOCHS_STAGE3 = 10   # Reduced from 20
    PSEUDO_WEIGHT = 0.5  # Weight for pseudo label loss

    # Inference
    THRESHOLD = 0.5
    MIN_AREA = 50
    TTA_SCALES = [1.0, 0.75, 1.25]

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cfg = Config()

# ========== GPU Optimization ==========
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# ========== Data Augmentation ==========
def get_train_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Perspective(scale=(0.02, 0.08), p=0.4),
        A.Affine(scale=(0.9, 1.1), rotate=(-5, 5), shear=(-5, 5), p=0.3),
        A.OneOf([
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8)),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.RandomGamma(gamma_limit=(80, 120)),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=3),
        ], p=0.2),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ========== Datasets ==========
class ReceiptDataset(Dataset):
    def __init__(self, img_dir, json_path, transform=None, is_test=False):
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)['images']
        self.image_names = list(self.data.keys())

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.img_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        if not self.is_test:
            mask = np.zeros((h, w), dtype=np.float32)
            words = self.data[img_name].get('words', {})
            for word_id, word_info in words.items():
                points = np.array(word_info['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], 1.0)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                return augmented['image'], augmented['mask']
            return image, mask
        else:
            if self.transform:
                augmented = self.transform(image=image)
                return augmented['image'], img_name, (h, w)
            return image, img_name, (h, w)


class PseudoLabelDataset(Dataset):
    """Dataset using generated pseudo labels"""
    def __init__(self, img_dirs, mask_dir, transform=None):
        self.transform = transform
        self.mask_dir = mask_dir
        self.samples = []

        for img_dir in img_dirs:
            if not os.path.exists(img_dir):
                continue
            for fname in os.listdir(img_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(img_dir, fname)
                    mask_name = Path(fname).stem + '.png'
                    mask_path = os.path.join(mask_dir, mask_name)
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))

        print(f"PseudoLabelDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            return augmented['image'], augmented['mask']
        return image, mask

# ========== Loss Functions ==========
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).clamp(1e-7, 1 - 1e-7)
        pt = torch.where(target == 1, pred, 1 - pred)
        alpha_t = torch.where(target == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        return (focal_weight * bce).mean()


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        pred_edge = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-8)

        target = target.unsqueeze(1) if target.dim() == 3 else target
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        target_edge = torch.sqrt(target_gx ** 2 + target_gy ** 2 + 1e-8)

        return F.mse_loss(pred_edge, target_edge)


class CombinedLoss(nn.Module):
    def __init__(self, dice_w=0.4, bce_w=0.3, focal_w=0.2, boundary_w=0.1):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='binary')
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = FocalLoss()
        self.boundary = BoundaryLoss()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.focal_w = focal_w
        self.boundary_w = boundary_w

    def forward(self, pred, target):
        target = target.unsqueeze(1) if target.dim() == 3 else target
        return (
            self.dice_w * self.dice(pred, target) +
            self.bce_w * self.bce(pred, target) +
            self.focal_w * self.focal(pred, target) +
            self.boundary_w * self.boundary(pred, target)
        )

# ========== Model ==========
class HighPrecisionOCRModel(nn.Module):
    def __init__(self, encoder_name, encoder_weights='imagenet'):
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
        out = self.base_model(x)
        return self.refine(out)

# ========== Post-processing ==========
def apply_morphology(mask, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def mask_to_polygons(mask, min_area=50, epsilon_factor=0.003):
    mask = apply_morphology((mask * 255).astype(np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 5 or h < 3:
            continue

        epsilon = epsilon_factor * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) >= 4:
            if len(approx) > 8:
                approx = cv2.convexHull(approx)
            polygons.append(approx.reshape(-1, 2).tolist())

    return polygons


def polygons_to_string(polygons):
    if not polygons:
        return ""
    return "|".join([" ".join([f"{int(p[0])} {int(p[1])}" for p in poly]) for poly in polygons])

# ========== Multi-scale TTA ==========
def multi_scale_tta(model, image, device, scales=[1.0, 0.75, 1.25], original_size=None):
    model.eval()
    _, h, w = image.shape
    all_preds = []

    with torch.no_grad():
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            new_h = max((new_h // 32) * 32, 32)
            new_w = max((new_w // 32) * 32, 32)

            scaled_img = F.interpolate(
                image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False
            ).to(device)

            with autocast('cuda', dtype=torch.bfloat16):
                pred = torch.sigmoid(model(scaled_img))
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
            all_preds.append(pred.float())

            with autocast('cuda', dtype=torch.bfloat16):
                pred_flip = torch.sigmoid(model(torch.flip(scaled_img, dims=[3])))
            pred_flip = torch.flip(pred_flip, dims=[3])
            pred_flip = F.interpolate(pred_flip, size=(h, w), mode='bilinear', align_corners=False)
            all_preds.append(pred_flip.float())

    final_pred = torch.stack(all_preds, dim=0).mean(dim=0)

    if original_size is not None:
        final_pred = F.interpolate(final_pred, size=original_size, mode='bilinear', align_corners=False)

    return final_pred.cpu().numpy()[0, 0]

# ========== Training Functions ==========
def train_epoch(model, loader, criterion, optimizer, scaler, device, accum_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Training")
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = masks.to(device, non_blocking=True).unsqueeze(1)

        with autocast('cuda', dtype=torch.bfloat16):
            preds = model(images)
            loss = criterion(preds, masks) / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        pbar.set_postfix({'loss': f'{total_loss / (i + 1):.4f}'})

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images = images.to(device, memory_format=torch.channels_last)
            masks = masks.to(device).unsqueeze(1)

            with autocast('cuda', dtype=torch.bfloat16):
                preds = model(images)
                loss = criterion(preds, masks)
            total_loss += loss.item()

    return total_loss / len(loader)


def compute_metrics(model, loader, device, threshold=0.5):
    model.eval()
    total_precision = total_recall = total_f1 = count = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Metrics"):
            images = images.to(device)
            masks = masks.numpy()

            with autocast('cuda', dtype=torch.bfloat16):
                preds = torch.sigmoid(model(images))
            preds = preds.float().cpu().numpy()

            for i in range(len(images)):
                pred = (preds[i, 0] > threshold).astype(float)
                gt = masks[i]

                tp = np.logical_and(pred, gt).sum()
                fp = np.logical_and(pred, ~gt.astype(bool)).sum()
                fn = np.logical_and(~pred.astype(bool), gt).sum()

                p = tp / (tp + fp + 1e-8)
                r = tp / (tp + fn + 1e-8)
                f1 = 2 * p * r / (p + r + 1e-8)

                total_precision += p
                total_recall += r
                total_f1 += f1
                count += 1

    return {
        'precision': total_precision / count,
        'recall': total_recall / count,
        'f1': total_f1 / count
    }

# ========== Stage Functions ==========
def stage1_train_base():
    """Stage 1: Train base model on original data"""
    print("=" * 60)
    print("STAGE 1: Training Base Model")
    print("=" * 60)

    torch.cuda.empty_cache()
    gc.collect()

    train_ds = ReceiptDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_JSON,
                              transform=get_train_transform(cfg.RESIZE_TARGET))
    val_ds = ReceiptDataset(cfg.VAL_IMG_DIR, cfg.VAL_JSON,
                            transform=get_val_transform(cfg.RESIZE_TARGET))

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = HighPrecisionOCRModel(cfg.ENCODER, cfg.ENCODER_WEIGHTS).to(cfg.DEVICE)
    model = model.to(memory_format=torch.channels_last)

    criterion = CombinedLoss().to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=0.01)

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=3)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS_STAGE1 - 3)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[3])

    scaler = GradScaler('cuda')
    best_loss = float('inf')

    for epoch in range(1, cfg.EPOCHS_STAGE1 + 1):
        print(f"\n--- Epoch {epoch}/{cfg.EPOCHS_STAGE1} ---")

        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 scaler, cfg.DEVICE, cfg.ACCUMULATION_STEPS)
        val_loss = validate(model, val_loader, criterion, cfg.DEVICE)

        print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'base_model.pth')
            print("Model saved!")

        if epoch % 10 == 0:
            metrics = compute_metrics(model, val_loader, cfg.DEVICE)
            print(f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

        scheduler.step()
        torch.cuda.empty_cache()

    print(f"\nStage 1 Complete! Best Val Loss: {best_loss:.4f}")


def stage2_generate_pseudo():
    """Stage 2: Generate pseudo labels for additional data"""
    print("=" * 60)
    print("STAGE 2: Generating Pseudo Labels")
    print("=" * 60)

    os.makedirs(cfg.PSEUDO_MASK_DIR, exist_ok=True)

    model = HighPrecisionOCRModel(cfg.ENCODER, None).to(cfg.DEVICE)
    model.load_state_dict(torch.load('base_model.pth'))
    model.eval()

    transform = get_val_transform(cfg.RESIZE_TARGET)

    for img_dir in cfg.PSEUDO_DIRS:
        if not os.path.exists(img_dir):
            print(f"Skip: {img_dir} not found")
            continue

        print(f"\nProcessing: {img_dir}")

        for fname in tqdm(os.listdir(img_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(img_dir, fname)
            image = cv2.imread(img_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = image.shape[:2]

            transformed = transform(image=image)
            img_tensor = transformed['image']

            mask = multi_scale_tta(model, img_tensor, cfg.DEVICE,
                                   scales=[1.0, 0.8], original_size=(orig_h, orig_w))

            mask_binary = (mask > cfg.THRESHOLD).astype(np.uint8) * 255

            mask_name = Path(fname).stem + '.png'
            cv2.imwrite(os.path.join(cfg.PSEUDO_MASK_DIR, mask_name), mask_binary)

    print(f"\nPseudo labels saved to: {cfg.PSEUDO_MASK_DIR}")


def stage3_train_with_pseudo():
    """Stage 3: Fine-tune with pseudo labels"""
    print("=" * 60)
    print("STAGE 3: Training with Pseudo Labels")
    print("=" * 60)

    torch.cuda.empty_cache()
    gc.collect()

    # Original data
    train_ds = ReceiptDataset(cfg.TRAIN_IMG_DIR, cfg.TRAIN_JSON,
                              transform=get_train_transform(cfg.RESIZE_TARGET))
    val_ds = ReceiptDataset(cfg.VAL_IMG_DIR, cfg.VAL_JSON,
                            transform=get_val_transform(cfg.RESIZE_TARGET))

    # Pseudo label data
    pseudo_ds = PseudoLabelDataset(cfg.PSEUDO_DIRS, cfg.PSEUDO_MASK_DIR,
                                   transform=get_train_transform(cfg.RESIZE_TARGET))

    # Combine datasets
    combined_ds = ConcatDataset([train_ds, pseudo_ds])
    print(f"Combined dataset: {len(combined_ds)} samples")

    train_loader = DataLoader(combined_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    model = HighPrecisionOCRModel(cfg.ENCODER, None).to(cfg.DEVICE)
    model.load_state_dict(torch.load('base_model.pth'))
    model = model.to(memory_format=torch.channels_last)

    criterion = CombinedLoss().to(cfg.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE * 0.5, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS_STAGE3)
    scaler = GradScaler('cuda')

    best_f1 = 0

    for epoch in range(1, cfg.EPOCHS_STAGE3 + 1):
        print(f"\n--- Epoch {epoch}/{cfg.EPOCHS_STAGE3} ---")

        train_loss = train_epoch(model, train_loader, criterion, optimizer,
                                 scaler, cfg.DEVICE, cfg.ACCUMULATION_STEPS)
        val_loss = validate(model, val_loader, criterion, cfg.DEVICE)
        metrics = compute_metrics(model, val_loader, cfg.DEVICE)

        print(f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        print(f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f} | F1: {metrics['f1']:.4f}")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), 'final_model.pth')
            print("Best F1 model saved!")

        scheduler.step()
        torch.cuda.empty_cache()

    print(f"\nStage 3 Complete! Best F1: {best_f1:.4f}")


def stage4_inference():
    """Stage 4: Final inference with multi-scale TTA"""
    print("=" * 60)
    print("STAGE 4: Final Inference")
    print("=" * 60)

    model = HighPrecisionOCRModel(cfg.ENCODER, None).to(cfg.DEVICE)

    model_path = 'final_model.pth' if os.path.exists('final_model.pth') else 'base_model.pth'
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Loaded: {model_path}")

    transform = get_val_transform(cfg.RESIZE_TARGET)

    with open(cfg.TEST_JSON, 'r', encoding='utf-8') as f:
        test_data = json.load(f)['images']

    preds = {}

    for img_name in tqdm(test_data.keys(), desc="Inference"):
        img_path = os.path.join(cfg.TEST_IMG_DIR, img_name)
        image = cv2.imread(img_path)
        if image is None:
            preds[img_name] = ""
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        transformed = transform(image=image)
        img_tensor = transformed['image']

        mask = multi_scale_tta(model, img_tensor, cfg.DEVICE,
                               scales=cfg.TTA_SCALES, original_size=(orig_h, orig_w))

        polygons = mask_to_polygons(mask > cfg.THRESHOLD, min_area=cfg.MIN_AREA)
        preds[img_name] = polygons_to_string(polygons)

    # Save submission
    sample_df = pd.read_csv(cfg.SAMPLE_SUB)
    sample_df['polygons'] = sample_df['filename'].map(preds).fillna("")

    output_csv = 'submission_high_precision.csv'
    sample_df.to_csv(output_csv, index=False)

    # Stats
    counts = sample_df['polygons'].apply(lambda x: len(x.split('|')) if x else 0)
    print(f"\nSaved: {output_csv}")
    print(f"Avg polygons: {counts.mean():.1f}")
    print(f"Max: {counts.max()}, Min: {counts.min()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2, 3, 4],
                        help='Training stage: 1=base, 2=pseudo, 3=finetune, 4=inference')
    args = parser.parse_args()

    if args.stage == 1:
        stage1_train_base()
    elif args.stage == 2:
        stage2_generate_pseudo()
    elif args.stage == 3:
        stage3_train_with_pseudo()
    elif args.stage == 4:
        stage4_inference()
