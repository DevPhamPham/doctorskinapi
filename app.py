from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
import cv2
import gdown

url = "https://drive.google.com/drive/folders/1QFPAAXkra2heEhN4ZwC6Xl54sLMS6Rxx?usp=sharing"
res = gdown.download_folder(url, quiet=False, use_cookies=False)
print("Downloading: ", res)

# print(tf.__version__)
app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return "hello world"

@app.route('/predict', methods=['POST'])
def predict():
    # đọc ảnh đầu vào từ request
    image = request.data
    try:
        # chuyển đổi ảnh 
        img_array = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
        img_resized = cv2.resize(img_array, (128, 128))
        # chuẩn hóa ảnh
        img_normalized = img_resized / 255.0
        # tải model
        model_path = os.path.abspath('./DataModel/mobilenetv1')
        print(model_path)
        model = tf.keras.models.load_model(model_path)

        # thực hiện dự đoán trên ảnh đầu vào
        result = model.predict(np.expand_dims(img_normalized, axis=0))
        
        # trả về kết quả dự đoán
        return jsonify({'result': result.tolist()})
    except Exception as e:
        print(e)
        return jsonify({'error': 'Something went wrong.','is': e})

# if __name__ == '__main__':
#     # Use the PORT environment variable provided by Heroku
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host='0.0.0.0', port=port)