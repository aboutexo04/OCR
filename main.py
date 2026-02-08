"""
FPN + ConvNeXt-Large OCR Text Detection
========================================
0.9553 달성 구조(smp.FPN + ConvNeXt-Base) 기반
- 백본: ConvNeXt-Base → ConvNeXt-Large
- EMA (Exponential Moving Average)
- Val 모니터링 + Best model 저장
- Train + Val 통합 학습
- Gradient Clipping + Accumulation
- 강화된 Augmentation
- 후처리: 0.9553 달성 설정 그대로 (UNCLIP=3.0, THRESH=0.3)
"""

import subprocess, sys, os

# Colab 환경: 자동 패키지 설치
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "segmentation-models-pytorch", "albumentations", "opencv-python-headless",
    "pyclipper", "shapely"])

import json, cv2, gc
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
import pyclipper
from shapely.geometry import Polygon
from tqdm import tqdm
import ssl, warnings

warnings.filterwarnings('ignore')
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Config ====================
ENCODER = "tu-convnext_large"   # Base → Large (파라미터 ~3배, 성능 대폭 향상)
SZ = 1024
BS = 4          # Large 모델이라 배치 줄임
ACCUM = 2       # 실효 배치 = 8 (기존과 동일)
EPOCHS = 25
LR = 5e-4       # Large 모델이라 LR 약간 낮춤
SHRINK_RATIO = 0.4
EMA_DECAY = 0.999

# 추론 파라미터 (0.9553 달성한 설정 그대로)
UNCLIP_RATIO = 3.0
BOX_THRESH = 0.3

BASE = './data/datasets'
PSEUDO = './data/pseudo_label'
MODEL_PATH = 'fpn_convnext_large_best.pth'
OUTPUT_CSV = 'submission_fpn_large.csv'


# ==================== Data Download ====================
def setup_data():
    if not os.path.exists(BASE):
        print("Downloading data...")
        os.system('wget -O data.tar.gz "https://aistages-api-public-prod.s3.amazonaws.com/app/Competitions/000377/data/data.tar.gz"')
        os.system('tar -xzf data.tar.gz')
    print(f"Data ready: {BASE}")


# ==================== Dataset ====================
class DBDataset(Dataset):
    def __init__(self, img_dir, json_path, transform=None, mode='train'):
        self.img_dir = img_dir
        self.transform = transform
        self.mode = mode
        self.data = {}
        self.image_names = []

        if json_path and os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)['images']
            self.image_names = list(self.data.keys())
        elif json_path:
            # train.json 없으면 val.json 시도
            alt = json_path.replace('train.json', 'val.json')
            if os.path.exists(alt):
                with open(alt, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)['images']
                self.image_names = list(self.data.keys())

    def __len__(self): return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        image = cv2.imread(os.path.join(self.img_dir, name))
        if image is None: return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)

        if self.mode != 'test' and 'words' in self.data[name]:
            for w_info in self.data[name]['words'].values():
                pts = np.array(w_info['points'], dtype=np.int32)
                try:
                    poly = Polygon(pts)
                    if poly.area > 0 and poly.length > 0:
                        d = poly.area * (1 - SHRINK_RATIO ** 2) / poly.length
                        pco = pyclipper.PyclipperOffset()
                        pco.AddPath(pts, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
                        shrunk = pco.Execute(-d)
                        if shrunk:
                            cv2.fillPoly(mask, [np.array(shrunk[0], dtype=np.int32)], 1)
                        else:
                            cv2.fillPoly(mask, [pts], 1)
                except:
                    cv2.fillPoly(mask, [pts], 1)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']

        if self.mode == 'test': return image, name, (h, w)
        return image, mask


# ==================== EMA ====================
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad: self.shadow[n] = p.data.clone()

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad: p.data.copy_(self.backup[n])
        self.backup = {}


# ==================== Augmentation ====================
train_tf = A.Compose([
    A.Resize(SZ, SZ),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
    A.Rotate(limit=20, p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.GaussNoise(p=0.15),
    A.Normalize(), ToTensorV2()
])
val_tf = A.Compose([A.Resize(SZ, SZ), A.Normalize(), ToTensorV2()])


# ==================== Training ====================
def train():
    torch.cuda.empty_cache(); gc.collect()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # --- Data: train + val + pseudo 전부 사용 ---
    datasets = []

    ds_train = DBDataset(os.path.join(BASE, 'images/train'),
                         os.path.join(BASE, 'jsons/train.json'), train_tf)
    if len(ds_train) > 0: datasets.append(ds_train)

    ds_val = DBDataset(os.path.join(BASE, 'images/val'),
                       os.path.join(BASE, 'jsons/val.json'), train_tf)
    if len(ds_val) > 0: datasets.append(ds_val)

    for folder in ['sroie', 'cord-v2', 'wildreceipt']:
        p_img = os.path.join(PSEUDO, folder, 'images')
        p_json = os.path.join(PSEUDO, folder, 'train.json')
        if os.path.exists(p_json) and os.path.exists(p_img):
            ds = DBDataset(p_img, p_json, train_tf)
            if len(ds) > 0: datasets.append(ds)

    full_ds = ConcatDataset(datasets)
    train_dl = DataLoader(full_ds, BS, shuffle=True, num_workers=4,
                          pin_memory=True, drop_last=True, persistent_workers=True)

    # Val loader (모니터링 전용)
    val_ds = DBDataset(os.path.join(BASE, 'images/val'),
                       os.path.join(BASE, 'jsons/val.json'), val_tf)
    val_dl = DataLoader(val_ds, BS, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(full_ds)}, Val monitor: {len(val_ds)}")

    # --- Model: 검증된 smp.FPN + 백본만 Large로 업그레이드 ---
    model = smp.FPN(
        encoder_name=ENCODER,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"FPN + {ENCODER}: {n_params:.1f}M params")

    # --- Optimizer (0.9553 모델과 동일 구조) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=EPOCHS,
        steps_per_epoch=len(train_dl) // ACCUM, pct_start=0.1)

    # Loss: 검증된 Dice + BCE 조합 (0.9553 모델과 동일)
    dice_fn = smp.losses.DiceLoss(mode='binary')
    bce_fn = smp.losses.SoftBCEWithLogitsLoss()
    scaler = GradScaler('cuda')
    ema = EMA(model, EMA_DECAY)

    best_val = float('inf')
    patience = 0

    print(f"=== Training: {SZ}x{SZ}, BS={BS}x{ACCUM}={BS * ACCUM}, Epochs={EPOCHS} ===")

    for ep in range(1, EPOCHS + 1):
        model.train()
        t_loss = 0
        optimizer.zero_grad(set_to_none=True)

        for i, (imgs, msks) in enumerate(tqdm(train_dl, desc=f"E{ep}/{EPOCHS}")):
            imgs = imgs.to(DEVICE, non_blocking=True)
            msks = msks.to(DEVICE, non_blocking=True).unsqueeze(1)

            with autocast('cuda', dtype=torch.bfloat16):
                preds = model(imgs)
                loss = (dice_fn(preds, msks) + bce_fn(preds, msks)) / ACCUM

            scaler.scale(loss).backward()

            if (i + 1) % ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                ema.update(model)

            t_loss += loss.item() * ACCUM

        # --- Val (EMA 모델로 평가) ---
        ema.apply(model)
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for imgs, msks in val_dl:
                imgs = imgs.to(DEVICE, non_blocking=True)
                msks = msks.to(DEVICE, non_blocking=True).unsqueeze(1)
                with autocast('cuda', dtype=torch.bfloat16):
                    preds = model(imgs)
                    v_loss += (dice_fn(preds, msks) + bce_fn(preds, msks)).item()

        avg_t = t_loss / len(train_dl)
        avg_v = v_loss / len(val_dl)
        print(f"E{ep}: train={avg_t:.4f} val={avg_v:.4f} lr={optimizer.param_groups[0]['lr']:.6f}")

        if avg_v < best_val:
            best_val = avg_v
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> Best model saved! (val={avg_v:.4f})")
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                print("  -> Early stopping")
                ema.restore(model)
                break

        ema.restore(model)
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  -> VRAM: {mem:.1f}GB")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    print(f"Training done. Best val={best_val:.4f}")
    return MODEL_PATH


# ==================== Inference ====================
def unclip(box, ratio):
    poly = Polygon(box)
    if poly.area <= 0 or poly.length <= 0:
        return [box.tolist() if hasattr(box, 'tolist') else box]
    d = poly.area * ratio / poly.length
    pco = pyclipper.PyclipperOffset()
    pco.AddPath([(int(p[0]), int(p[1])) for p in box],
                pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = pco.Execute(d)
    return expanded if expanded else [box.tolist() if hasattr(box, 'tolist') else box]


def inference(model_path, box_thresh=BOX_THRESH, unclip_ratio=UNCLIP_RATIO,
              output_csv=OUTPUT_CSV):
    print(f"\n=== Inference (thresh={box_thresh}, unclip={unclip_ratio}) ===")

    model = smp.FPN(encoder_name=ENCODER, in_channels=3, classes=1).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_ds = DBDataset(os.path.join(BASE, 'images/test'),
                        os.path.join(BASE, 'jsons/test.json'), val_tf, mode='test')
    test_dl = DataLoader(test_ds, BS, shuffle=False, num_workers=4)

    results = {}
    with torch.no_grad():
        for imgs, names, (ohs, ows) in tqdm(test_dl, desc="Inference"):
            imgs = imgs.to(DEVICE)
            with autocast('cuda', dtype=torch.bfloat16):
                p1 = torch.sigmoid(model(imgs))
                p2 = torch.sigmoid(model(torch.flip(imgs, [3])))
                preds = (p1 + torch.flip(p2, [3])) / 2
            preds = preds.float().cpu().numpy()

            for i, name in enumerate(names):
                oh, ow = ohs[i].item(), ows[i].item()
                mask = cv2.resize(preds[i][0], (ow, oh))
                binary = (mask > box_thresh).astype(np.uint8)

                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                polys = []
                for cnt in contours:
                    if cv2.contourArea(cnt) < 30: continue
                    eps = 0.003 * cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, eps, True)
                    if len(approx) < 3: continue
                    pts = approx.reshape(-1, 2)
                    try:
                        expanded = unclip(pts, unclip_ratio)
                        final = np.array(expanded[0])
                        polys.append(final.reshape(-1).tolist())
                    except:
                        polys.append(pts.reshape(-1).tolist())

                results[name] = "|".join(" ".join(map(str, p)) for p in polys)

    df = pd.read_csv(os.path.join(BASE, 'sample_submission.csv'))
    df['polygons'] = df['filename'].map(results).fillna("")
    df.to_csv(output_csv, index=False)

    counts = df['polygons'].apply(lambda x: len(x.split('|')) if x else 0)
    print(f"Saved: {output_csv}")
    print(f"Avg poly/img: {counts.mean():.1f}, Min: {counts.min()}, Max: {counts.max()}")


# ==================== Main ====================
if __name__ == "__main__":
    setup_data()
    saved = train()

    # 기본 추론 (0.9553 달성 설정)
    inference(saved, box_thresh=0.3, unclip_ratio=3.0,
              output_csv='submission_fpn_large.csv')

    # 파라미터 튜닝 추론 (학습 없이 바로 실행)
    inference(saved, box_thresh=0.25, unclip_ratio=3.0,
              output_csv='submission_fpn_large_v2.csv')
    inference(saved, box_thresh=0.2, unclip_ratio=3.5,
              output_csv='submission_fpn_large_v3.csv')
