import tensorflow.keras  # tensorflow.keras 모듈을 import합니다. 

import numpy as np  # numpy 모듈을 import합니다. 
import cv2  # opencv 모듈을 import합니다.

# keras_model.h5 모델을 로드합니다.
model = tensorflow.keras.models.load_model('keras_model.h5')

# 0번 카메라(기본 캠 카메라)를 캡처하는 VideoCapture 객체를 생성합니다.
cap = cv2.VideoCapture(0)

# 입력 이미지의 크기를 설정합니다.
size = (224, 224)

# 이미지 분류에 대한 클래스 라벨을 설정합니다.
classes = ['GunBbang', 'prechell', 'crowsando', 'diget', 'chocopie']

while cap.isOpened():
    # 카메라로부터 프레임을 읽어옵니다.
    ret, img = cap.read()

    # 프레임 읽기가 실패하면 종료합니다.
    if not ret:
        break

    # 이미지의 높이, 너비, 채널 수를 가져옵니다.
    h, w, _ = img.shape

    # 이미지에서 중앙 부분을 추출합니다.
    cx = h / 2
    img = img[:, 200:200+img.shape[0]]

    # 이미지를 좌우로 뒤집습니다.
    img = cv2.flip(img, 1)

    # 이미지를 모델 입력 크기로 조정합니다.
    img_input = cv2.resize(img, size)

    # 이미지를 RGB에서 BGR로 변환합니다.
    img_input = cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB)

    # 이미지를 [-1, 1] 범위로 정규화합니다.
    img_input = (img_input.astype(np.float32) / 127.0) - 1

    # 모델에 대한 입력으로 이미지를 4D 배열로 확장합니다.
    img_input = np.expand_dims(img_input, axis=0)

    # 모델을 사용하여 이미지를 예측합니다.
    prediction = model.predict(img_input)

    # 예측 결과에서 가장 높은 값을 갖는 인덱스를 가져옵니다.
    idx = np.argmax(prediction)

    # 예측 결과를 이미지에 텍스트로 표시합니다.
    cv2.putText(img, text=classes[idx], org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(255, 255, 255), thickness=2)

    # 결과 이미지를 화면에 보여줍니다.
    cv2.imshow('result', img)

    # 'q' 키를 누르면 종료합니다.
    if cv2.waitKey(1) == ord('q'):
        break
