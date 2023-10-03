import copy
import time
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import random_split, DataLoader

from custom_models import CustomMobileNetV3Large, QuantizedCustomMobileNetV3Large, CustomMnasNet1_3, QuantizedCustomMnasNet1_3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# 모델학습 헬퍼 함수
def train_models(model, dataloaders, criterion, optimizer, num_epochs=10, is_inception=False, quantize=False):

    torch.cuda.empty_cache()

    val_acc_history = []
    val_top5_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    patience = 3
    no_improvement_count = 0

    since = time.time()
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1} / {num_epochs}')
        print('-' * 10)

        # 각 에폭은 학습/검증 phase를 가짐
        for phase in ['train', 'val']:
            print(f'>>>>> phase: {phase}<<<<<')
            if phase == 'train':
                model.train()  # 트레이닝 모드 설정
            else:
                if quantize == True:
                    # 모델을 양자화 + 추론모드
                    model = torch.quantization.convert(model.eval(), inplace=False)
                else:
                    model.eval()  # 추론 모드 설정

            running_loss = 0.0
            running_corrects = 0
            running_top5_corrects = 0

            # 데이터 학습하기
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 파라미터 기울기 초기화
                optimizer.zero_grad()

                # forward
                # 학습 모드인 경우에만 history 추적
                with torch.set_grad_enabled(phase == 'train'):
                    # 모델의 ouptut과 loss를 구함
                    # inception의 경우 학습시 auxiliary output이 있는 특수 케이스임.
                    #   학습시: final output과 auxiliary output을 더하는 과정이 필요함
                    #   테스트시: final output만 고려
                    # Auxilary output을 같이 고려해야하는 학습단계
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # top-5 정확도 계산
                    _,top5_preds = torch.topk(outputs, 5)
                    top5_corrects = torch.sum(top5_preds == labels.view(-1, 1))

                    # 학습(backward + optimize)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (epoch + 1) % 5 == 0:
                            # 모델 및 다른 정보 저장
                            save_path = f'save/{type(model).__name__}_epoch{epoch+1}_quntize({quantize}).pth'
                            torch.save({
                                'model_state_dict': model.state_dict(),  # 모델 가중치 저장
                                'optimizer_state_dict': optimizer.state_dict(),  # 옵티마이저 상태 저장 (선택적)
                                'epoch': epoch+1,  # 현재 학습 에폭 저장 (선택적)
                                # 다른 필요한 정보 저장 (선택적)
                            }, save_path)

                # loss 구하기
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_top5_corrects += top5_corrects

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_top5_acc = running_top5_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 모델 깊은복사
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                no_improvement_count = 0  # 개선이 있었으므로 카운트 초기화
            if phase == 'val':
                val_acc_history.append(epoch_acc.item())
                val_top5_acc_history.append(epoch_top5_acc.item())
                no_improvement_count += 1  # 개선이 없었으므로 카운트

            # Early Stopping 확인
            if no_improvement_count >= patience:
                print(f"No improvement in validation accuracy for {patience} epochs. Early stopping...")
                break  # 학습 종료

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    print('Best top5 val Acc: {:.4f}'.format(epoch_top5_acc))

    # 베스트 모델 가중치를 로드
    model.load_state_dict(best_model_wts)
    return model, val_acc_history, val_top5_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True, quantize=False):
  # 모델마다 다르게 지정될 변수들 초기화
  model_ft = None
  input_size = 0

  if model_name == 'resnet':
    if quantize:
      '''Quantized Resnet50'''
      weights = models.quantization.ResNet50_QuantizedWeights.DEFAULT
      model_ft = models.quantization.resnet50(weights=weights, quantize=True)
    else:
      '''Resnet50'''
      model_ft = models.resnet50(pretrained=use_pretrained)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    set_parameter_requires_grad(model_ft, feature_extract)
    input_size = 224

  elif model_name == 'mobilenet':
    if quantize:
      '''Quantized Mobilenet v3 large'''
      model_ft = QuantizedCustomMobileNetV3Large(num_classes=500).to(device)

      # 양자화 설정 - gbgemm 백엔드 사용
      backend = 'fbgemm'
      # 양자화 스키마 설정 (예: symmetric)
      quantization_params = torch.quantization.get_default_qconfig(backend)
      quantization_params = torch.quantization.QConfig(activation=quantization_params.activation,
                                                       weight=quantization_params.weight)

      # 양자화 준비
      model.qconfig = quantization_params
      model_ft = torch.quantization.prepare_qat(model_ft, inplace=False)

    else:
      model_ft = CustomMobileNetV3Large(num_classes=500).to(device)
    set_parameter_requires_grad(model_ft, feature_extract)
    input_size = 224

  elif model_name == 'mnasnet':
    if quantize:
      '''Quantized Mnasnet 1_3'''
      model_ft = QuantizedCustomMnasNet1_3(num_classes=500).to(device)

      # 양자화 설정 - gbgemm 백엔드 사용
      backend = 'fbgemm'
      model_ft.qconfig = torch.quantization.get_default_qat_qconfig(backend)

      # 양자화 준비
      model_ft = torch.quantization.prepare_qat(model_ft, inplace=False)

    else:
      model_ft = models.mnasnet1_3(weights=models.MNASNet1_3_Weights.IMAGENET1K_V1)
      num_ftrs = model_ft.classifier[1].in_features
      model_ft.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
      # model_ft = CustomMnasNet1_3(num_classes=500).to(device)
    set_parameter_requires_grad(model_ft, feature_extract)
    input_size = 224

  elif model_name == 'efficientnet':
    if quantize:
      pass
    else:
      model_ft = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
      num_ftrs = model_ft.classifier[1].in_features
      model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes, bias=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    input_size = 224

  elif model_name == 'shufflenet':
    if quantize:
      pass
    else:
      model_ft = models.shufflenet_v2_x2_0(weights=models.ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1)
      num_ftrs = model_ft.fc.in_features
      model_ft.fc = nn.Linear(num_ftrs, num_classes, bias=True)
    set_parameter_requires_grad(model_ft, feature_extract)
    input_size = 224

  elif model_name == 'alexnet':
    '''AlexNet'''
    model_ft = models.alexnet(pretrianed=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_featrues
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    input_size = 224

  elif model_name == 'vgg':
    '''VGG11_bn'''
    model_ft = models.vgg11_bn(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
    input_size = 224

  elif model_name == 'squeezenet':
    '''Squeezenet 1.0'''
    model_ft = models.squeezenet1_0(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
    model_ft.num_classes = num_classes
    input_size = 224

  elif model_name == 'densenet':
    '''Densenet 121'''
    model_ft = models.densenet121(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)
    input_size = 224

  elif model_name == 'inception':
    '''Inception V3'''
    if quantize:
      weights = models.quantization.Inception_V3_QuantizedWeights.DEFAULT
      model_ft = models.quantization.inception_v3(weights=weights, quantize=True)
    else:
      model_ft = models.inception_v3(pretrained=use_pretrained)
    set_parameter_requires_grad(model_ft, feature_extract)
    # Auxilary net
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 299  # 다른 모델과 다르게 299 사이즈를 사용
  else:
    print('모델의 이름을 잘못 입력하여 종료합니다...')
    exit()

  return model_ft.to(device), input_size
