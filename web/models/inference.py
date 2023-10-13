from time import time
import torch
from torchvision import models
import torch.nn as nn

from custom_models import CustomMobileNetV3Large
from get_dataloader import get_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference(model, weight_path, batch_size, is_grayscale=False):
  # 데이터 불러오기
  since = time()
  test_dir = r'D:/data/training/sources/test'
  test_loader = get_dataloader(test_dir, batch_size, is_train=False, is_grayscale=is_grayscale)

  # 모델 로드
  num_classes = 500
  if model == 'mobilenet':
    model = CustomMobileNetV3Large(num_classes)
  elif model == 'efficientnet':
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes, bias=True)
  elif model == 'shufflenet':
    model = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes, bias=True)
  else:
    print('모델 이름을 확인하세요')

  checkpoint = torch.load(weight_path)
  model.load_state_dict(checkpoint['model_state_dict'])  # 모델 가중치 불러오기
  model.to(device)

  with torch.no_grad():
    correct = 0
    total = 0
    top5_correct = 0

    for batch_idx, (inputs, labels) in enumerate(test_loader):
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      # top-1 정확도
      _, predicted = torch.max(outputs, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

      # top-5 정확도 계산
      _, top5_preds = torch.topk(outputs, 5)
      top5_correct += sum(labels.view(-1, 1).eq(top5_preds)).sum().item()

  top1_accuracy = round(100 * correct / total, 2)
  top5_accuracy = round(100 * top5_correct / total, 2)

  accuracy_dict = {
    'top1': top1_accuracy,
    'top5': top5_accuracy}

  latency = round(time()-since, 2)

  print(f'Test Top-1 Accuracy: {accuracy_dict["top1"]}')
  print(f'Test Top-5 Accuracy: {accuracy_dict["top5"]}')
  print(f'Latency: {latency}')

  return predicted, accuracy_dict


if __name__ == '__main__':
  # prediction, accuracy = inference(weight_path='save/mobilenet_epoch10_batch128_pretrained_Augmentated.pth',
  #                                  is_grayscale=False)
  # print('=' * 20)
  #
  # accuracy_grayscale = inference(weight_path='mobilenet_epoch10_batch128_pretrained_Augmentated_Grayscaled.pth',
  #                                is_grayscale=True)

  # 4개 모델 latency 비교
  print('===== MobileNet =====')
  model = 'mobilenet'
  weight_path = 'save/mobilenet_epoch10_batch128_pretrained_Augmentated.pth'
  inference(model, weight_path, batch_size=126, is_grayscale=False)

  print('===== EfficientNet =====')
  model = 'efficientnet'
  weight_path = 'save/EfficientNet_epoch5_quntize(False).pth'
  inference(model, weight_path, batch_size=126, is_grayscale=False)

  print('===== ShuffleNet =====')
  model = 'shufflenet'
  weight_path = 'save/ShuffleNetV2_epoch5_quntize(False).pth'
  inference(model, weight_path, batch_size=126, is_grayscale=False)
