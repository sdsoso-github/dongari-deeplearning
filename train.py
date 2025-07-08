import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os 

# 폴더 위치 지정
dataset_dir = "dataset"

# 이미지 크기와 한 번에 학습할 이미지 개수 설정
IMG_SIZE = (128, 128)  # 이미지를 128x128로 크기 조정
BATCH_SIZE = 32  # 한 번에 32장씩 모델에 전달

# 훈련용 이미지를 불러올 때, 이미지 변형(증강)을 적용하는 설정
train_datagen = ImageDataGenerator(
    rescale=1./255,  # 이미지 픽셀값을 0~255 → 0~1 사이로 정규화 (딥러닝에 적합하도록)
    validation_split=0.2,  # 전체 이미지 중 20%는 검증용으로 사용
    rotation_range=20,  # 이미지를 최대 20도까지 회전시켜 학습 데이터 다양화
    width_shift_range=0.2,  # 이미지를 좌우로 조금 이동
    height_shift_range=0.2,  # 이미지를 위아래로 조금 이동
    shear_range=0.15,  # 이미지 변형
    zoom_range=0.1,  # 이미지를 확대하거나 축소
    horizontal_flip=True,  # 이미지를 좌우 반전시켜서 다양한 경우 학습
    fill_mode='nearest'  # 이동 시 생기는 빈 공간을 주변 픽셀로 채움
)

# 실제 훈련용 이미지를 가져오는 제너레이터 만들기
train_generator = train_datagen.flow_from_directory(
    dataset_dir,  # 고양이/개 폴더가 들어있는 상위 폴더 경로
    target_size=IMG_SIZE,  # 이미지를 128x128 크기로 변경
    batch_size=BATCH_SIZE,  # 한 번에 불러올 이미지 수
    class_mode='binary',  # 고양이/개 이진 분류
    subset='training'  # 전체 데이터 중 훈련용 데이터만 사용
)

# 검증용 이미지를 가져오는 제너레이터 만들기
val_generator = train_datagen.flow_from_directory(
    dataset_dir,  # 위와 동일한 폴더
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'  # 전체 데이터 중 검증용 데이터만 사용
)

# 딥러닝 모델 만들기 (CNN 구조)
model = tf.keras.Sequential([  # 순차적으로 층을 쌓는 방식

    tf.keras.Input(shape=(128, 128, 3)),  # 입력 이미지 크기 (128x128, RGB 3채널)

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),  # 32개의 필터로 특징 추출 (3x3 크기)
    tf.keras.layers.MaxPooling2D(2,2),  # 특징 맵의 크기를 절반으로 줄임

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),  # 더 많은 필터로 더 복잡한 특징 추출
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),  # 더 깊은 필터
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),  # 2D 데이터를 1D로 펴줌 (완전 연결층에 넣기 위해)
    tf.keras.layers.Dense(256, activation='relu'),  # 은닉층: 뉴런 256개
    tf.keras.layers.Dropout(0.5),  # 일부 뉴런을 랜덤으로 끄면서 과적합 방지
    tf.keras.layers.Dense(1, activation='sigmoid')  # 출력층: 0~1 사이 확률 출력 (이진 분류)
])

# 모델 학습 설정
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy',  # 이진 분류 문제에 적합한 손실 함수
    metrics=['accuracy']  # 모델 평가 지표로 정확도 사용
)

# 모델 구조 요약 출력 (층별 구조 확인 가능)
model.summary()

# 모델을 실제로 학습시킴
model.fit(
    train_generator,  # 훈련 데이터 넣기
    validation_data=val_generator,  # 검증 데이터도 같이 넣기
    epochs=1000
)

# 학습이 끝난 모델을 저장
model.save("models/cat_dog_model.keras")
