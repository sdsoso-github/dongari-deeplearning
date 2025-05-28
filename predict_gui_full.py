import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image

# 모델 로딩
model = tf.keras.models.load_model("models/cat_dog_model.keras")
IMG_SIZE = (128, 128)

# 예측 함수
def predict_image(img_path):
    img = keras_image.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras_image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    label = "개" if pred > 0.5 else "고양이"
    confidence = pred if pred > 0.5 else 1 - pred
    return label, confidence

# 이미지 열기 및 예측 함수
def open_and_predict():
    file_path = filedialog.askopenfilename(
        title="이미지 파일 선택",
        filetypes=[("이미지 파일", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return

    # 이미지 미리보기
    img = Image.open(file_path)
    img = img.resize((200, 200))
    img_tk = ImageTk.PhotoImage(img)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    # 예측
    label, conf = predict_image(file_path)
    messagebox.showinfo("예측 결과", f"{label}입니다!\n확률: {conf:.2%}")

# GUI 구성
root = tk.Tk()
root.title("고양이 vs 개 이미지 분류기")
root.geometry("300x350")

select_button = tk.Button(root, text="이미지 선택", command=open_and_predict, font=("Arial", 14))
select_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

note = tk.Label(root, text="이미지를 선택하면 예측 결과가 팝업됩니다", font=("Arial", 10))
note.pack(pady=5)

root.mainloop()
