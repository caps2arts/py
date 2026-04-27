from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import re

app = Flask(__name__)
CORS(app)  # разрешаем запросы с GitHub Pages

def count_caps(image_bytes):
    """Считает крышки на фото через анализ круглых форм и цвета"""
    # Декодируем изображение
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return 0

    # Уменьшаем для скорости если фото большое
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    # Конвертируем в серый
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Ищем круги через HoughCircles
    # minRadius и maxRadius подобраны под крышки от бутылок
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,        # минимальное расстояние между центрами
        param1=60,         # порог для детектора Canny
        param2=35,         # порог накопителя (чем меньше — тем больше находит)
        minRadius=15,      # минимальный радиус крышки в пикселях
        maxRadius=80       # максимальный радиус крышки в пикселях
    )

    if circles is None:
        return 0

    count = len(circles[0])
    return count


@app.route('/count', methods=['POST'])
def count():
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({'error': 'Нет изображения', 'count': 0}), 400

        # Изображение приходит как base64
        image_data = data['image']

        # Убираем заголовок data:image/...;base64,
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        count = count_caps(image_bytes)

        return jsonify({'count': count, 'success': True})

    except Exception as e:
        return jsonify({'error': str(e), 'count': 0}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
