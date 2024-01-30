- https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/

# 1. Experiments
## 1) Data Augmentation
- 다음과 같은 Data augmentation을 사용했을 때 Training loss가 0.3 후반대에서 그 밑으로 떨어지지 않았습니다.
    ```python
    self.transformer = A.Compose(
        [
            A.Flip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.3,
                scale_limit=0,
                rotate_limit=180,
                border_mode=cv2.BORDER_WRAP,
                p=1,
            ),
            A.Normalize(mean=img_mean, std=img_std),
            ToTensorV2(),
        ]
    )
    ```
- 다음과 같이 무작위 좌우 또는 상하 이미지 뒤집기와 이미지 회전을 제외했을 때 Training loss가 더 작은 값까지 감소할 수 있었으며 Validation loss도 더 낮은 최소값을 기록했습니다.
    ```python
    self.transformer = A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=0.3,
                scale_limit=0,
                rotate_limit=0,
                border_mode=cv2.BORDER_WRAP,
                p=1,
            ),
            A.Normalize(mean=img_mean, std=img_std),
            ToTensorV2(),
        ]
    )
    ```
## 2) Incidence Angle
- Training set에 대해서 결측치가 아닌 Incidence angle을 시각화했습니다.
    - <img src="https://github.com/KimRass/KimRass/assets/67457712/ed416fd3-8dd4-4169-aec9-1451a1b1396b" width="400">
- Incidence angle이 특정한 값을 가질 때는 그에 해당하는 이미지의 Label이 반드시 Iceberg가 됨을 볼 수 있습니다.
- 따라서 Incidence angle을 Feature로서 사용했습니다.
- VGG16의 Architecture를 조금 변경했습니다. 이미지와 Incidence angle을 합쳐 4개의 채널을 가지는 텐서를 입력으로 받도록 했고 마지막 FC layer의 출력 차원을 2로 변경했습니다.
## CutMix & Label Smoothing
- CutMix only: Val loss 0.2000.
- Label smoothing only: Val loss 0.4390.
- Both: 정확히 측정해보지는 않았으나 Label smoothing만 사용했을 때와 비슷한 수준.
## 3) Adaptive Instance Normalization
$$\text{AdaIN}(x, y) = \sigma(y)\bigg(\frac{x - \mu(x)}{\sigma(x)}\bigg) + \mu(y)$$
