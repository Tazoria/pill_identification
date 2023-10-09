import json
import pickle
import cv2
import numpy as np
import torch
import glob
from torchvision.transforms import transforms

from web.models.custom_models import CustomMobileNetV3Large

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


def home(request):
  return render(request, 'index.html')


def get_model():
  model = CustomMobileNetV3Large(num_classes=500)
  checkpoint = torch.load('web/models/save/CustomMobileNetV3Large_epoch5_quntize(False).pth')
  model.load_state_dict(checkpoint['model_state_dict'])

  return model


# def predict(image_tensor):
#   model = get_model()
#   outputs = model(image_tensor)
#   _, prediction = torch.max(outputs, 1)
#   with open('web/models/save/classes.pkl', 'rb') as pickle_file:
#     classes = pickle.load(pickle_file)
#     prediction = classes[prediction.item()]
#
#   return prediction
def predict(image_tensor):
  model = get_model().eval()
  with torch.no_grad():
    outputs = model(image_tensor)
    _, top5 = torch.topk(outputs, k=5, dim=1)  # 상위 5개 예측
    top1 = top5[:, 0].item()
    print('='*10)
    print('top5', top5)
    print('top1', top1)
  #   top1 = predicted[:, 0]
  with open('web/models/save/classes_batch128.pkl', 'rb') as pickle_file:
    classes = pickle.load(pickle_file)
    print('classes > ', classes[:5])
    prediction = classes[top1]
  print('prediction', prediction)
  print('='*10)

  return prediction


@csrf_exempt  # CSRF 검증에서 제외 - 외부 클라이언트로부터 POST 요청을 받을 때 CSRF 토큰을 확인하지 않음
def upload(request):
  if request.method == 'POST' and request.FILES['imageInput']:
    image = request.FILES['imageInput']

    image_data = image.read()
    image_np = np.frombuffer(image_data, np.uint8)
    image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # 이미지를 파이토치 텐서로 변환
    image_tensor = torch.from_numpy(image_cv2).permute(2, 0, 1).float()
    # 배치 차원 추가
    image_tensor = image_tensor.unsqueeze(0)

    data_transforms = transforms.Compose([
      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 이미지를 정규화하고 ToTensor 변환을 수행
    image_tensor = data_transforms(image_tensor)

    # 추론
    prediction = predict(image_tensor)
    json_path = glob.glob(f'D:/data/training/labels/extracted_all/{prediction}_json/*.json')[0]

    with open(json_path, 'r', encoding='utf-8') as pill_data:
      pill_data = json.load(pill_data)
      pill_name = pill_data['images'][0]['dl_name']
      pill_image = pill_data['images'][0]['img_key']
      pill_company = pill_data['images'][0]['dl_company']
      pill_class = pill_data['images'][0]['di_class_no']

      response_data = {'prediction': prediction,
                       'pill_name': pill_name,
                       'pill_company': pill_company,
                       'pill_class': pill_class,
                       'pill_image': pill_image}
      print(f'===== {prediction} 추론 완료 =====')
      return JsonResponse(response_data)

  return render(request, 'index.html')









