import os
import json
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

# --- 1. 환경 설정 ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_IMG_DIR = 'data/datasets/images/train'
TRAIN_JSON = 'data/datasets/jsons/train.json'
TEST_IMG_DIR = 'data/datasets/images/test'
MODEL_PATH = 'best_model.pth'

# --- 2. 데이터셋 정의 ---
class ReceiptDataset(Dataset):
    def __init__(self, root_dir, json_path=None, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        if is_train:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)['images']
            self.image_names = list(self.data.keys())
        else:
            self.image_names = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        if self.is_train:
            mask = np.zeros((h, w), dtype=np.float32)
            words = self.data[img_name].get('words', {})
            for word_id in words:
                points = np.array(words[word_id]['points'], dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                return augmented['image'], augmented['mask']
        else:
            if self.transform:
                augmented = self.transform(image=image)
                return augmented['image'], img_name, (h, w)
        
        return image

# --- 3. 전처리 및 증강 ---
train_transform = A.Compose([
    A.Resize(640, 640),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(640, 640),
    A.Normalize(),
    ToTensorV2()
])

# --- 4. 메인 실행 함수 (학습 + 추론) ---
def main():
    # A. 데이터 준비
    train_ds = ReceiptDataset(TRAIN_IMG_DIR, TRAIN_JSON, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)

    # B. 모델 선언
    model = smp.UnetPlusPlus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = smp.losses.DiceLoss(mode='binary')

    # C. 학습 (간략화)
    print("Starting Training...")
    model.train()
    for epoch in range(1, 6): # 빠른 확인을 위해 5에폭만 설정
        for images, masks in tqdm(train_loader):
            images, masks = images.to(DEVICE), masks.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), MODEL_PATH)

    # D. 추론 및 제출 파일 생성 (UFO 형식 또는 CSV)
    print("Starting Inference...")
    model.eval()
    test_ds = ReceiptDataset(TEST_IMG_DIR, transform=test_transform, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    
    results = {}

    with torch.no_grad():
        for img, img_name, (orig_h, orig_w) in tqdm(test_loader):
            img_name = img_name[0]
            output = model(img.to(DEVICE))
            pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            
            # 원본 사이즈로 복구
            pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
            # 컨투어 추출 (Polygon 좌표화)
            contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            words_dict = {}
            for i, cnt in enumerate(contours):
                if len(cnt) < 3: continue # 삼각형 미만 무시
                points = cnt.reshape(-1, 2).tolist()
                words_dict[f"{i+1:04d}"] = {"points": points}
            
            results[img_name] = {"words": words_dict}

    # E. JSON 결과 저장 (대회 표준 UFO 형식)
    output_json = {"images": results}
    with open('submission.json', 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=4)
    print("Inference Done! 'submission.json' saved.")

if __name__ == "__main__":
    # 데이터가 없다면 먼저 wget 실행 권장
    if not os.path.exists(TRAIN_IMG_DIR):
        print("Data not found! Please run wget and tar commands first.")
    else:
        main()