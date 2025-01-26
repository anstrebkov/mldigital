import io
import pickle
import numpy as np
import PIL.Image
import PIL.ImageOps
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging

# Настройка логов
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели
try:
    with open('mnist_model.pkl', 'rb') as f:
        model = pickle.load(f)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Создание FastAPI приложения
app = FastAPI()

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все домены (для разработки)
    allow_credentials=True,
    allow_methods=["*"],  # Разрешить все методы (GET, POST, etc.)
    allow_headers=["*"],  # Разрешить все заголовки
)

# Эндпоинт для предсказания
@app.post("/predict-image/")
async def predict_image(file: UploadFile = File(...)):
    try:
        logger.info("Received image for prediction")
        contents = await file.read()
        
        # Преобразование изображения
        pil_image = PIL.Image.open(io.BytesIO(contents)).convert('L')  # Конвертация в grayscale
        pil_image = PIL.ImageOps.invert(pil_image)  # Инверсия цветов
        pil_image = pil_image.resize((28, 28), PIL.Image.ANTIALIAS)  # Изменение размера
        img_array = np.array(pil_image).reshape(1, -1)  # Преобразование в массив
        
        # Предсказание
        prediction = model.predict(img_array)
        logger.info(f"Prediction: {prediction[0]}")
        
        return {"prediction": int(prediction[0])}
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail="Error processing image")

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)