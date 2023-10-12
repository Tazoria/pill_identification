from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_transform(is_train=True, is_grayscale=False):
  # 데이터 변환: 데이터가 부족해 과적합이 발생한 것으로 보이고 색상에 민감해 보여 흑백 + 학습데이터 증강

  # 학습 데이터 변환시
  if is_train:

    # 그레이스케일 적용시
    if is_grayscale:
      transform = transforms.Compose([
        transforms.RandAugment(num_ops=4, magnitude=9, fill=[128]),  # 적용할 변환의 수, 강도, 여백채우기 색깔
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),  # 고정된 크기로 이미지 크기 조정
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize([0.485], [0.229])  # Grayscale 정규화
      ])
    # 그레이스케일 미적용시
    else:
      transform = transforms.Compose([
        transforms.RandAugment(num_ops=4, magnitude=9, fill=[128]),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # RGB 정규화
      ])

  # 검증 / 테스트 데이터 변환시
  else:
    if is_grayscale:
      transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),  # 이미지 중심부 자르기
        transforms.ToTensor(),  # 이미지를 텐서로 변환
        transforms.Normalize([0.485], [0.229])  # Grayscale 정규화
      ])
    else:
      transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ])

  return transform


def get_dataloader(data_dir, batch_size=512, is_train=False, is_grayscale=False):
  if is_train:
    # 데이터 변환 설정
    transform = get_transform(is_train=True, is_grayscale=is_grayscale)
    # 데이터셋 생성
    dataset = ImageFolder(data_dir, transform=transform)
    # 데이터로더 생성
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
  else:
    # 데이터 변환 설정
    transform = get_transform(is_train=False, is_grayscale=is_grayscale)
    # 데이터셋 생성
    dataset = ImageFolder(data_dir, transform=transform)
    # 데이터 로더 생성
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

  return dataloader


if __name__ == '__main__':
  # get_samples()
  # get_crop()

  test_loader = get_dataloader(data_dir=r'D:/data/training/sources/test',
                               batch_size=128, is_train=False, is_grayscale=False)
  print(test_loader)

  # 데이터로더가 잘 생성됐는지 보기
  for images, labels in test_loader:
    for i in range(5):
      image = images[i]
      label = labels[i]
      print(f"이미지 형태: {image.shape}")
      print(f"이미지 라벨: {label}")
