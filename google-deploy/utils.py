import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, load_img
from tensorflow.keras.preprocessing.image import img_to_array
from distance import L2_siamese_dist

def transform_image(byte_image):
    image = load_img(byte_image)
    # Convert to array
    array = img_to_array(image).astype('uint8')
    #image = np.asarray(image)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(array,1.1,6)
    if len(faces) != 1:
      #print('wajah tidak ditemukan')
      return -1
    x,y,w,h = faces.squeeze()
    crop = image[y:y+h,x:x+w]

    # Resizing the image to be 105x105
    image = tf.image.resize(crop, (105,105))
    # Scaling the image to be range 0 and 1
    image /= 255.0
    # Fix the shape of the image that will be fed to the model
    #[1, 105, 105, 3]
    image = np.expand_dims(image, axis=0)
    return image

model = load_model(
    'my_model_v2.h5',
    custom_objects={'L2_siamese_dist':L2_siamese_dist}
)

def predict_person(input_image_url, verification_urls, recognition_threshold=0.5, verification_threshold=0.6):
    inp_img = transform_image(input_image_url)
    # Result predictions
    results = []
    for verification_url in verification_urls:
        val_img = transform_image(verification_url)
        result = model.predict([inp_img, val_img])
        results.append(result)

    recognition = np.sum(np.array(results) > recognition_threshold)
    verification = recognition / len(verification_urls)
    verified = verification >= verification_threshold

    return verified
