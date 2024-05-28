# 음향 데이터 기반 기계 고장 진단 AI 모델 개발

# 소개
기계 유지 관리는 작업 효율성을 보장하고 예상치 못한 고장을 방지하는 데 매우 중요합니다. 전통적인 유지 관리 방법은 시간과 비용이 많이 들 수 있습니다. 이 프로젝트에서는 AI를 활용하여 기계의 음향 데이터를 분석하여 잠재적인 결함을 조기에 감지할 수 있습니다.


# 특징
- 음원에서 데이터 분석, 특징 추출
- 원시 음향 데이터 전처리
- 기계 학습 모델의 훈련 및 평가
- 결함 분류 및 예측
- 결과 시각화
- 이상 탐지를 위한 AutoEncoder


# 탐색적 데이터 분석(EDA)
탐색적 데이터 분석(EDA)은 데이터세트의 특성을 이해하는 데 중요한 단계입니다. 이 프로젝트에서는 Time Domain과 Frequency Domain 모두의 신호를 분석하고, STFT(Short-Time Fourier Transform)를 사용하여 자세한 주파수 분석을 수행합니다.


## Time Domain 분석
>Time Domain 분석에는 원시 음향 신호를 시간의 함수로 검사하여 전체 sampling rate가 16000HZ이고, 1277개의 10초 단위의 신호 데이터가 있음을 확인하였습니다.

## Frequency Domain 분석
>Frequency Domain 분석은 시간 영역 신호를 주파수 영역으로 변환하여 신호의 주파수 구성 요소를 드러냅니다. 이는 푸리에 변환을 사용하여 수행됩니다. 결함 진단에 중요할 수 있는 기본 주파수 구성요소와 강도를 표시합니다. 전체 train_dataset에서 최대 주파수와 최소 주파수를 확인하였습니다. 
![FFTMAxMIn](https://github.com/JiyeongHEO/DaconAcoustics/blob/main/fft.jpg)


## 상세한 분석을 위한 STFT
>STFT(Short-Time Fourier Transform)는 시간에 지남에 따라 신호의 주파수 성분이 어떻게 변하는지 정보를 제공합니다. 고주파 대역의 신호에서 고장 신호가 존재 할 것으로 예측하였습니다.
![STFT Sample](https://github.com/JiyeongHEO/DaconAcoustics/blob/main/stft.jpg)


## 데이터 전처리
>Stft로 분석한 신호를 Min-Max Scale로 정규화하여 학습 데이터를 생성,
데이터 증강(Data Augmentation)을 통해 제한된 Dataset의 다양성을 증가시켰습니다.

# Model Training
## 모델 선택
>Convolutional AutoEncoder를 선택해 기계의 정상적인 신호를 학습하고 그로부터의 편차를 식별하여 이상 탐지를 할 수 있게 하였습니다.

```
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # 인코더 정의
        self.encoder = nn.Sequential(
            nn.Conv2d(3, k, kernel_size=3, stride=2, padding=1),  # 3x128x128 -> 16x64x64
            nn.ReLU(),
            nn.Conv2d(k, 2*k, kernel_size=3, stride=2, padding=1),  # 16x64x64 -> 32x32x32
            nn.ReLU(),
            nn.Conv2d(2*k, 4*k, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.Conv2d(4*k, 8*k, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # 디코더 정의
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8*k, 4*k, kernel_size=2, stride=2), 
            nn.ConvTranspose2d(4*k, 2*k, kernel_size=2, stride=2),
            nn.ConvTranspose2d(2*k, k, kernel_size=2, stride=2),  # 32x32x32 -> 16x64x64
            nn.ConvTranspose2d(k, 3, kernel_size=2, stride=2),  # 16x64x64 -> 3x128x128
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
```



## 훈련(Training)
>훈련 데이터와 검증 데이터는 과적합을 완화하고 K-Fold Cross-Validation를 적용하여 분리합니다.
손실 함수로 실제값과 예측값 사이의 평균 제곱 차이를 계산해 더 큰 오류를 loss값으로 하는 MSE(Mean Squared Error)를 사용하고, 최적화로 Adam을 사용합니다.

```
# 학습 데이터, 검증 데이터 분리
kf = KFold(n_splits=5, shuffle=True)
```

```
# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
# criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=lr)
```

## Validation 
    별도의 검증 세트에서 모델 성능을 검증합니다.



