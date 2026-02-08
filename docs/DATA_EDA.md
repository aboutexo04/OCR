# OCR 대회 데이터 EDA (Exploratory Data Analysis)

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 대회명 | AI Stages OCR Competition (000377) |
| 과제 | 영수증 이미지에서 텍스트 영역(Word-level) 검출 |
| 데이터 출처 | drp.en_ko.in_house.selectstar |
| 언어 | 한국어(ko), 영어(en), 기타(others) |
| 총 데이터 용량 | 약 3.4GB |

---

## 2. 폴더 구조

```
OCR/
├── data/                                    # 전체 데이터 (3.4GB)
│   ├── datasets/                            # 주요 데이터셋
│   │   ├── images/
│   │   │   ├── train/                       # 학습 이미지 (3,272개)
│   │   │   ├── val/                         # 검증 이미지 (404개)
│   │   │   └── test/                        # 테스트 이미지 (413개)
│   │   ├── jsons/
│   │   │   ├── val.json                     # 검증 라벨 (31.8MB)
│   │   │   └── test.json                    # 테스트 메타 (58KB)
│   │   └── sample_submission.csv            # 제출 샘플
│   │
│   └── pseudo_label/                        # 추가 학습용 Pseudo Label 데이터
│       ├── sroie/
│       │   └── images/
│       │       ├── train/                   # 556개
│       │       └── test/                    # 360개
│       ├── wildreceipt/
│       │   └── images/                      # 1,768개
│       └── cord-v2/
│           └── images/                      # 1,000개
│
├── experiments/                             # 실험 결과 저장용
├── 20260128_OCR_baseline.ipynb              # 베이스라인 노트북 (UNet++ + ResNet34)
├── 20160128_OCR_9129_8622_9718.ipynb        # 개선 노트북 (EfficientNet-b3)
├── main.py                                  # 학습/추론 스크립트
├── submission*.csv                          # 제출 파일들
└── .gitignore
```

---

## 3. 데이터셋 통계

### 3.1 이미지 분포

| 데이터셋 | 이미지 수 | 비고 |
|----------|-----------|------|
| Train | 3,272 | 학습용 |
| Validation | 404 | 검증용 (라벨 있음) |
| Test | 413 | 평가용 (라벨 없음) |
| **총합** | **4,089** | - |

### 3.2 Pseudo Label 데이터 (추가 학습용)

| 데이터셋 | 이미지 수 | 형식 |
|----------|-----------|------|
| SROIE (train) | 556 | jpg |
| SROIE (test) | 360 | jpg |
| WildReceipt | 1,768 | jpeg |
| CORD-v2 | 1,000 | png |
| **총합** | **3,684** | - |

---

## 4. 이미지 분석

### 4.1 이미지 크기 분포 (Train 샘플 기준)

| 항목 | Width | Height |
|------|-------|--------|
| 최소값 | 335px | 609px |
| 최대값 | 1,280px | 1,280px |
| 평균값 | 932px | 1,190px |

### 4.2 이미지 방향

| 방향 | 비율 | 개수 (50개 샘플) |
|------|------|------------------|
| 세로 (Portrait) | 66.0% | 33개 |
| 가로 (Landscape) | 34.0% | 17개 |

### 4.3 파일 크기

| 항목 | 값 |
|------|-----|
| 최소 | 53 KB |
| 최대 | 333 KB |
| 평균 | 144 KB |

### 4.4 파일명 패턴

```
drp.en_ko.in_house.selectstar_XXXXXX.jpg
```
- 형식: `drp.{언어}.{출처}.{제공자}_{6자리 번호}.jpg`
- 예시: `drp.en_ko.in_house.selectstar_000003.jpg`

---

## 5. 라벨 분석 (val.json 기준)

### 5.1 Word 통계

| 항목 | 값 |
|------|-----|
| 총 이미지 수 | 404개 |
| 총 Word 수 | 46,714개 |
| 이미지당 평균 Word 수 | **115.63개** |
| 최소 Word 수 | 48개 |
| 최대 Word 수 | 276개 |

### 5.2 Orientation (텍스트 방향) 분포

| Orientation | 개수 | 비율 |
|-------------|------|------|
| Horizontal | 42,912 | 91.9% |
| None | 3,793 | 8.1% |
| (기타) | 9 | 0.0% |

### 5.3 Language (언어) 분포

| Language | 개수 | 비율 |
|----------|------|------|
| 한국어 (ko) | 26,260 | **90.5%** |
| 영어 (en) | 2,756 | 9.5% |
| 기타 (others) | 11 | 0.0% |

> **주요 인사이트**: 데이터의 90% 이상이 한국어이며, 대부분 수평 방향 텍스트

### 5.4 Word 바운딩 박스 크기

| 항목 | Width | Height |
|------|-------|--------|
| 최소값 | 2.7px | 2.2px |
| 최대값 | 957.7px | 746.4px |
| 평균값 | **81.5px** | **24.6px** |

### 5.5 Aspect Ratio (가로/세로 비율)

| 항목 | 값 |
|------|-----|
| 최소 | 0.10 |
| 최대 | 99.66 |
| 평균 | **3.77** |

> **주요 인사이트**: 평균 Aspect Ratio 3.77은 텍스트가 일반적으로 가로로 긴 형태임을 의미

---

## 6. JSON 데이터 구조

### 6.1 val.json / train.json 구조

```json
{
    "images": {
        "이미지명.jpg": {
            "words": {
                "0001": {
                    "points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
                    "orientation": "Horizontal",
                    "language": ["ko"]
                },
                "0002": { ... }
            },
            "img_w": 1106,
            "img_h": 1280
        }
    }
}
```

### 6.2 test.json 구조 (라벨 없음)

```json
{
    "images": {
        "이미지명.jpg": {
            "words": {},
            "img_w": 1106,
            "img_h": 1280
        }
    }
}
```

### 6.3 Points 좌표 시스템

- 4개의 꼭짓점 좌표 (Quadrilateral)
- 순서: 좌상단 → 우상단 → 우하단 → 좌하단 (시계방향)
- 단위: 픽셀 (float)

---

## 7. 제출 형식 분석

### 7.1 sample_submission.csv 구조

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| filename | string | 이미지 파일명 |
| polygons | string | 폴리곤 좌표 문자열 |

### 7.2 Polygon 포맷

```
x1 y1 x2 y2 x3 y3 x4 y4|x1 y1 x2 y2 x3 y3 x4 y4|...
```

- 각 폴리곤: 공백으로 구분된 x y 좌표쌍
- 폴리곤 간 구분: `|` (파이프)
- 좌표: 정수형

### 7.3 제출 파일 비교

| 파일명 | 평균 폴리곤 수 | 최대 폴리곤 수 |
|--------|---------------|---------------|
| submission.csv (baseline) | 29.7 | 81 |
| submission_effv2_onecycle.csv | 47.3 | 103 |
| submission_clahe_fixed.csv | 45.0 | 123 |
| submission_multiscale_final.csv | **49.8** | 107 |

> **인사이트**: val.json 기준 이미지당 평균 115개 Word가 있으므로, 현재 모델들은 약 40~50% 정도만 검출 중

---

## 8. 모델 아키텍처 (노트북 기준)

### 8.1 Baseline (20260128_OCR_baseline.ipynb)

| 항목 | 설정 |
|------|------|
| 모델 | UNet++ |
| Encoder | ResNet34 (ImageNet pretrained) |
| 입력 크기 | 512×512 |
| 배치 크기 | 16 (Accumulation: 2) |
| Loss | Dice Loss |
| Optimizer | AdamW (lr=1e-4) |
| Scheduler | CosineAnnealing |
| Epochs | 20 |

### 8.2 개선 버전 (20160128_OCR_9129_8622_9718.ipynb)

| 항목 | 설정 |
|------|------|
| 모델 | UNet++ |
| Encoder | **EfficientNet-b3** |
| 입력 크기 | **1024×1024** |
| 배치 크기 | 4 (Accumulation: 8) |
| Loss | **BCE + Dice (Combined)** |
| 증강 | Perspective, Flip, Rotate90, GaussNoise, MotionBlur |
| TTA | 원본 + 좌우반전 앙상블 |
| Epochs | 30 |

---

## 9. 데이터 특성 요약

### 9.1 강점
- 고해상도 이미지 (평균 932×1190)
- 풍부한 라벨 정보 (Word 단위 폴리곤 + 언어/방향 메타데이터)
- Pseudo Label 데이터로 데이터 증강 가능

### 9.2 도전 과제
- 작은 텍스트 검출 (평균 Word 크기 81×24px)
- 다양한 Aspect Ratio (0.1 ~ 99.66)
- 현재 모델 검출률 약 40-50% (개선 필요)
- train.json 파일 부재 (val.json만 라벨 존재)

### 9.3 권장 개선 방향
1. **고해상도 입력**: 1024×1024 이상 권장
2. **멀티스케일 접근**: 작은 텍스트와 큰 텍스트 동시 처리
3. **Pseudo Label 활용**: SROIE, WildReceipt, CORD-v2 데이터 추가 학습
4. **후처리 개선**: 폴리곤 단순화 파라미터 조정 (epsilon 값)
5. **앙상블**: 다양한 threshold와 TTA 조합

---

## 10. 참고 파일

| 파일 | 용도 |
|------|------|
| main.py | 기본 학습/추론 스크립트 |
| 20260128_OCR_baseline.ipynb | 베이스라인 코드 |
| 20160128_OCR_9129_8622_9718.ipynb | 개선 버전 코드 |
| .gitignore | Git 제외 파일 설정 |

---

*Generated: 2026-02-05*
