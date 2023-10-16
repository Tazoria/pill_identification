import json
import pickle
from PIL import Image
import torch
import glob
from torchvision.transforms import transforms

from web.models.custom_models import CustomMobileNetV3Large

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def home(request):
  return render(request, 'index.html')


def get_model():
  # 모델 및 다른 정보 불러오기
  load_path = 'web/models/save/[Mobilenet] Mobilenet_RandAugment(126).pth'
  checkpoint = torch.load(load_path)
  model = CustomMobileNetV3Large(num_classes=500)
  model.load_state_dict(checkpoint['model_state_dict'])  # 모델 가중치 불러오기
  print(f'Model loaded from {load_path}')

  model.to(device)
  model.eval()

  return model


def preprocess(file_path):
  image = Image.open(file_path)

  # 사진을 모델 입력에 맞게 resize
  image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ])
  image_tensor = image_transform(image).unsqueeze(0)

  return image_tensor
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
  model = get_model()
  with torch.no_grad():
    outputs = model(image_tensor.to(device))
    _, top5 = torch.topk(outputs, k=5, dim=1)  # 상위 5개 예측
    top1 = top5[:, 0].item()
    print('='*10)
    print('top5', top5)
    print('top1', top1)
  #   top1 = predicted[:, 0]
  with open('web/models/save/classes_batch128.pkl', 'rb') as pickle_file:
    classes = pickle.load(pickle_file)
    prediction = classes[top1]
    top5_prediction = []
    for idx in top5[0]:
      top5_prediction.append(classes[idx])
  print('prediction', prediction)
  print('top5_prediction', top5_prediction)
  print('='*10)

  print(prediction, top5_prediction)

  return prediction, top5_prediction


@csrf_exempt  # CSRF 검증에서 제외 - 외부 클라이언트로부터 POST 요청을 받을 때 CSRF 토큰을 확인하지 않음
def upload(request):
  if request.method == 'POST' and request.FILES['imageInput']:
    file = request.FILES['imageInput']

    upload_path = 'web/static/img/uploaded_img.jpg'
    fp = open(upload_path, 'wb')
    for chunk in file.chunks():
      fp.write(chunk)
    fp.close()
    image_tensor = preprocess(upload_path)

    # 추론
    prediction, top5_prediction = predict(image_tensor)
    json_path = glob.glob(f'D:/data/training/labels/extracted_all/{prediction}_json/*.json')[0]

    with open(json_path, 'r', encoding='utf-8') as pill_data:
      pill_data = json.load(pill_data)
      pill_name = pill_data['images'][0]['dl_name']
      pill_company = pill_data['images'][0]['dl_company']
      pill_class = pill_data['images'][0]['di_class_no']
      output_image = pill_data['images'][0]['img_key']

    response_data = {'prediction': prediction,
                     'pill_name': pill_name,
                     'pill_company': pill_company,
                     'pill_class': pill_class,
                     'top5_prediction': top5_prediction,
                     'input_image': 'http://127.0.0.1:8000/static/img/uploaded_img.jpg',
                     'output_image': output_image
                     }

    print(f'===== {prediction} 추론 완료 =====')
    return JsonResponse(response_data)
  return render(request, 'index.html')









