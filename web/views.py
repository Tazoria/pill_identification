import cv2
import numpy as np
import torch

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt


def home(request):
  return render(request, 'index.html')


@csrf_exempt  # CSRF 검증에서 제외 - 외부 클라이언트로부터 POST 요청을 받을 때 CSRF 토큰을 확인하지 않음
def upload_image(request):
  if request.method == 'POST' and request.FILES['imageInput']:
    image = request.FILES['imageInput']

    # PIL.Image 객체로 변환
    image_np = np.asarray(bytearray(image), dtype=np.uint8)
    image_cv2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    # 이미지를 파이토치 텐서로 변환
    image_tensor = torch.from_numpy(image_cv2).permute(2, 0, 1).float() / 255.0
    # 배치 차원 추가
    image_tensor = image_tensor.unsqueeze(0)








