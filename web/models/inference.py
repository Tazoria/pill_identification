from tqdm import tqdm
import torch
from custom_models import CustomMobileNetV3Large
from get_dataloader import get_test_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference(model=CustomMobileNetV3Large(num_classes=500), is_grayscale=False):
  # 데이터 불러오기
  test_dir = r'D:/data/training/sources/test'

  # 모델 로드
  if is_grayscale:
    checkpoint = torch.load('save/mobilenet_epoch10_batch128_pretrained_Augmentated_Grayscaled.pth')
    test_loader = get_test_loader(test_dir, batch_size=126, is_train=False, is_grayscale=is_grayscale)
  else:
    test_loader = get_test_loader(test_dir, batch_size=126, is_train=False, is_grayscale=is_grayscale)
    checkpoint = torch.load('save/mobilenet_epoch10_batch128_pretrained_Augmentated.pth')
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
      top5_correct += sum(labels.view(-1, 1) == predicted).sum().item()

  top1_accuracy = round(100 * correct / total, 2)
  top5_accuracy = round(100 * top5_correct / total, 2)

  accuracy_dict = {
    'top1': top1_accuracy,
    'top5': top5_accuracy}

  print(f'Test Top-1 Accuracy: {accuracy_dict["top1"]}')
  print(f'Test Top-5 Accuracy: {accuracy_dict["top5"]}')

  return accuracy_dict


if __name__ == '__main__':
  accuracy_dict = inference(is_grayscale=False)
  print('=' * 20)
  accuracy_dict_grayscale = inference(is_grayscale=True)
