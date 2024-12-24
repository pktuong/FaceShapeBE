from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import tensorflow as tf

# print(tf.__version__)

app = FastAPI()

# # CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load model
model = tf.keras.models.load_model(
    'vgg16-face-1.h5'
)
# y_label_dict = {0: 'Heart', 1: 'Oblong', 2: 'Oval', 3: 'Round', 4: 'Square'}

# def preprocess_image(image):
#     img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
#     img = cv2.resize(img, (224, 224))
#     img = img / 255.0  # Normalize
#     img = np.expand_dims(img, axis=0)
#     return img

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         image = await file.read()
#         processed_img = preprocess_image(image)
#         prediction = model.predict(processed_img)
#         label_index = np.argmax(prediction)
#         result = y_label_dict[label_index]
#         confidence = np.max(prediction) * 100
#         return {"shape": result, "confidence": round(confidence, 2)}
#     except Exception as e:
#         return {"error": str(e)}

# @app.get("/test")
# async def test():
#     return {"message": "Hello World"}