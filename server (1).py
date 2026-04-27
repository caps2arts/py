from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io

app = Flask(__name__)
CORS(app)

def count_caps(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return 0

    # Уменьшаем если фото большое
    h, w = img.shape[:2]
    if w > 1200:
        scale = 1200 / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=25,
        param1=50,
        param2=30,
        minRadius=12,
        maxRadius=70
    )

    if circles is None:
        return 0

    return len(circles[0])


@app.route('/count', methods=['POST'])
def count():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'Нет изображения', 'count': 0}), 400

        image_data = data['image']
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        result = count_caps(image_bytes)

        return jsonify({'count': result, 'success': True, 'manual': False})

    except Exception as e:
        return jsonify({'error': str(e), 'count': 0, 'manual': True}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
