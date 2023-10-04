import torch
import torchvision
from web.models.custom_models import CustomMobileNetV3Large


def infer(image_tensor):
    # 모델 불러와 적용하기
    model = CustomMobileNetV3Large(num_classes=500)
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)

    checkpoint = torch.load('models/save/mobilenet_sample.pth')
    model.load_state_dict(checkpoint['model_state_dict'])  # 모델 가중치 불러오기

    # 모델을 평가 모드로 설정
    model.eval()
    outputs = model(image_tensor)
    values, indices = torch.topk(outputs, k=5)
    predicted_label = values.item()

    # 원래 클래스 이름 아는방법 찾기
    # ImageFolder의 클래스 레이블과 클래스 이름 간의 매핑 정보 확인
    class_to_idx = dataset.class_to_idx

    # 클래스 레이블을 클래스 이름으로 변환
    predicted_classname = list(class_to_idx.keys())[list(class_to_idx.values()).index(predicted_label)]




    correct = 0
    total = 0
    with torch.no_grad():
      for inputs, labels in tqdm_notebook(test_dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy}')