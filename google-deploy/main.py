import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import numpy as np
from urllib.request import urlopen
from distance import L2_siamese_dist
from flask import Flask, request, jsonify
import warnings
warnings.filterwarnings("ignore")
from flask_cors import CORS


model = tf.keras.models.load_model(
    'my_model_v2.h5',
    custom_objects={'L2_siamese_dist':L2_siamese_dist}
)


def get(url):
    with urlopen(str(url.numpy().decode("utf-8"))) as request:
        img_array = np.asarray(bytearray(request.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_image_from_url(url):
    return tf.py_function(get, [url], tf.uint8)


def transform_image(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image,1.1,6)
    if len(faces) != 1:
      return -1
    x,y,w,h = faces.squeeze()
    crop = image[y:y+h,x:x+w]

    # Resizing the image to be 105x105
    image = tf.image.resize(crop, (105,105))
    # Scaling the image to be range 0 and 1
    image /= 255.0
    # Fix the shape of the image that will be fed to the model
    #(1, 105, 105, 3)
    image = np.expand_dims(image, axis=0)
    return image

def predict(input_image_url, verification_urls, recognition_threshold=0.5, verification_threshold=0.6):
    inp_img = transform_image(input_image_url)
    # Result predictions
    results = []
    for verification_url in verification_urls:
        val_img = transform_image(verification_url)
        result = model.predict([inp_img, val_img])
        results.append(result)
  
    recognition = np.sum(np.array(results) > recognition_threshold)
    verification = recognition / len(verification_urls)
    verified = verification > verification_threshold

    return verified

app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        response = request.get_json()
        image_urls = response.get('urls')
        image_urls = image_urls.split()
        if image_urls is None or len(image_urls) < 3:
            return jsonify({'error': 'not enough file'})
        
        try:
            dataset = tf.data.Dataset.from_tensor_slices(image_urls)
            dataset = dataset.map(lambda x: read_image_from_url(x)).cache()
            
            input_image_array = list(dataset.as_numpy_iterator())[-1]
            verification_array = list(dataset.as_numpy_iterator())[:-1]
            pred = predict(input_image_array, verification_array)
            result = ""
            if pred:
                result = 'verified'
            else:
                result = 'unknown'
            return jsonify({'pred': result})
            
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get("PORT",8080)))
