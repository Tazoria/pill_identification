import torch
import torch.nn as nn
from torchvision import models


class CustomMobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(CustomMobileNetV3Large, self).__init__()
        self.features = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, progress=True).features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, num_classes, 1, 1, 0),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 양자화를 고려한 학습: MobileNetV3 Large
class QuantizedCustomMobileNetV3Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(QuantizedCustomMobileNetV3Large, self).__init__()
        '''
            [ 양자화 ]
            - 모델의 가중치와 활성화 값을 낮은 비트 수로 표현하여 모델을 경량화하고 속도를 향상시키는 기술
            - Stub: 모델 내에서 양자화 연산을 수행하기 전과 후에 데이터의 변환을 지원하는 역할
            - QuantStub(): 모델의 입력 데이터를 양자화된 형태로 변환하고, 양자화된 형태에서 연산을 수행
            - DeQuantStugb(): 양자화된 출력을 다시 원래 형태의 실수 값으로 변환
        '''
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.features = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT, progress=True).features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(960, num_classes, 1, 1, 0),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.quant(x)  # 양자화
        x = self.features(x)
        x = self.classifier(x)
        x = self.dequant(x)  # 되돌리기
        return x


# class CustomMnasNet1_3(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(CustomMnasNet1_3, self).__init__()
#         # MNASNet의 Convolutional 레이어를 추출하여 features로 설정
#         self.features = models.mnasnet1_3(weights=models.MNASNet1_3_Weights.DEFAULT, progress=True).layers
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.2, inplace=True),
#             nn.Linear(in_features=1280, out_features=num_classes, bias=True)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x


class CustomMnasNet1_3(models.MNASNet):
    def __init__(self, num_classes=1000):
        super(CustomMnasNet1_3, self).__init__(alpha=1)

        # 기본 MNASNet 모델의 classifier 부분을 새로운 classifier로 교체
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )


class QuantizedCustomMnasNet1_3(nn.Module):
    def __init__(self, num_classes=1000):
        super(QuantizedCustomMnasNet1_3, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        # MNASNet의 Convolutional 레이어를 추출하여 features로 설정
        self.features = nn.Sequential(*list(models.mnasnet1_3(weights=models.MNASNet1_3_Weights.DEFAULT, progress=True).children())[:-1])  # 마지막 Fully Connected 레이어 제외
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x