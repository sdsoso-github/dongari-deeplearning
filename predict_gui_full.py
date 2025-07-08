import tkinter as tk  
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk 
import tensorflow as tf
import numpy as np 
from tensorflow.keras.preprocessing import image as keras_image  

# 미리 학습해둔 모델(.keras 파일)을 불러오기
model = tf.keras.models.load_model("models/cat_dog_model.keras")

# 모델에 넣기 위한 이미지 크기 설정
IMG_SIZE = (128, 128)

def predict_image(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)  # 이미지를 128x128 크기로 불러오기
    img_array = keras_image.img_to_array(img) / 255.0  # 이미지를 숫자 배열로 바꾸고 0~1 범위로 정규화
    img_array = np.expand_dims(img_array, axis=0)  # (1, 128, 128, 3) 모양으로 차원 추가 (모델 입력 형태 맞추기)
    
    pred = model.predict(img_array)[0][0]  # 모델이 예측한 값 (0~1 사이 확률)을 가져옴
    
    # 예측한 값이 0.5보다 크면 "개", 작으면 "고양이"로 판단
    label = "개" if pred > 0.5 else "고양이"
    # 확률 계산: 예측한 값이 높을수록 해당 클래스일 가능성 ↑
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence  # 예측 결과와 신뢰도 반환

def open_and_predict():
    # 사용자에게 파일을 선택하라고 요청하는 창 띄우기
    file_path = filedialog.askopenfilename(
        title="이미지 파일 선택",
        filetypes=[("이미지 파일", "*.jpg *.jpeg *.png")]  # 선택 가능한 파일 확장자
    )
    
    if not file_path:
        return  # 아무 파일도 선택하지 않으면 그냥 끝냄

    # 선택한 이미지를 열고 크기를 줄여서 미리보기용으로 변환
    img = Image.open(file_path)
    img = img.resize((200, 200))  # tkinter 창에 맞게 200x200으로 축소
    img_tk = ImageTk.PhotoImage(img)  # tkinter에서 쓸 수 있는 이미지 형태로 바꾸기

    image_label.configure(image=img_tk)  # 이미지 보여줄 라벨에 이미지 넣기
    image_label.image = img_tk  # 참조를 저장해서 이미지가 사라지지 않도록 함

    # 예측 함수 실행해서 결과 받아오기
    label, conf = predict_image(file_path)

    # 팝업창으로 예측 결과 출력
    messagebox.showinfo("예측 결과", f"{label}입니다!\n확률: {conf:.2%}")


# tkinter 메인 윈도우 생성
root = tk.Tk()
root.title("고양이 vs 개 이미지 분류기")  # 창 제목
root.geometry("300x350")  # 창 크기 (가로 300, 세로 350)

# "이미지 선택" 버튼 생성
select_button = tk.Button(root, text="이미지 선택", command=open_and_predict, font=("Arial", 14))
select_button.pack(pady=10)  # 위쪽에 여백 10 추가

# 이미지를 보여줄 공간(라벨) 만들기
image_label = tk.Label(root)
image_label.pack(pady=10)

# 설명 문구 보여주기
note = tk.Label(root, text="이미지를 선택하면 예측 결과가 팝업됩니다", font=("Arial", 10))
note.pack(pady=5)

root.mainloop()
