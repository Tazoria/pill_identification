import os
import json
from time import time
import glob
from tqdm import tqdm
import zipfile
import cv2
from send2trash import send2trash
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# zip파일을 압축해제하는 함수
def extraction(path):
  since = time()
  zip_paths = glob.glob(path + f'/*.zip')
  extract_path = path + '/extracted'
  print('======Extracting =====')
  for zip_path in tqdm(zip_paths):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      zip_ref.extractall(extract_path)
    os.remove(zip_path)
  time_elapsed = time() - since
  print('압축해제 완료 > {:.2f} 소요'.format(time_elapsed))


# 이미지 파일을 크롭후 크롭폴더로 이동하고 남은 빈 원본 디렉토리 삭제
def rmdir_empty(path):
  is_file = os.listdir(path)
  if len(is_file) == 0:
    os.rmdir(path)
    print('dir removed > ', path.split('\\')[-1])


# 크롭
def crop(image_paths, crop_path):

    # 각 이미지파일마다 반복
    for img_path in tqdm(image_paths):
        # json파일이 저장된 상위 디렉토리
        dir_name = img_path.split('\\')[-2]
        dir_path = f'D:/data/training/sources/extracted/{dir_name}'
        # 파일 이름(확장자 X)
        file_name = img_path.split('\\')[-1].split('.png')[0]
        json_path = f'D:/data/training/labels/extracted/{dir_name}_json/{file_name}.json'
        if os.path.isfile(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

                # bbox 정보에 접근
                x0 = json_data['annotations'][0]['bbox'][0]  # x_min
                y0 = json_data['annotations'][0]['bbox'][1]  # y_min
                x1 = x0 + json_data['annotations'][0]['bbox'][2]  # x_min + width
                y1 = y0 + json_data['annotations'][0]['bbox'][3]  # y_min + height

            # 이미지 크롭 및 저장
            src = cv2.imread(img_path)  # 이미지를 크롭하기 위해 읽어오기
            src = src[y0:y1, x0:x1]  # 이미지를 bbox정보에 따라 크롭(cv2: H X W X C)
            src = cv2.resize(src, (229, 229))
            # # 크롭이 잘 됐는지 테스트 해보기 - 잘 됨
            # src_rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            # plt.axis('off')
            # plt.imshow(src_rgb)

            # 이미지를 저장할 디렉토리가 미리 생성되어 있어야지 저장됨
            # ImageFolder를 사용해 데이터셋을 생성할 것이므로 폴더를 생성
            crop_file_path = crop_path + f'/{dir_name}'

            if not os.path.isdir(crop_file_path):
                os.mkdir(crop_file_path)

            cropped_file_name = crop_file_path + f'/{file_name}.jpg'
            result = cv2.imwrite(cropped_file_name, src)  # 크롭된 이미지를 png->jpg 저장해 용량을 더욱 줄임

            if not result:
                print('저장 실패!!')
                break
            os.remove(img_path)
            # json파일에서 bbox정보를 가져오기위해 파일을 열고 읽어들이기
            rmdir_empty(dir_path)

        else:
          # json 파일이 없는 경우 기록
          with open('no_json.txt', 'a') as f:
            f.write(f'{file_name}.json\n')
          # 없는 경우 이미지 파일과 디렉토리를 함께 해당 경로로 이동 후 원래 디렉토리는 삭제
          destination_dir = f'D:/data/training/sources/no_json'
          if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
          os.rename(dir_path+f'/{file_name}.png', destination_dir+f'/{file_name}.jpg')
          os.rmdir(dir_path)

    print('\n>>>>>>>>>> 파일 크롭 완료 <<<<<<<<<<\n')


if __name__ == '__main__':
  source_path = 'D:/data/training/sources'
  label_path = 'D:/data/training/labels'
  img_paths = glob.glob('D:/data/training/sources/extracted/*/*.png')
  crop_paths = 'D:/data/training/sources/cropped'

  # 압축해제
  # extraction(label_path)
  # extraction(source_path)
  crop(img_paths, crop_paths)
