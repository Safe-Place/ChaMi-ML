import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from utils import predict
#from flask_cors import CORS

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

app = Flask(__name__)
#CORS(app)

@app.route("/")
def hello_world():
    name = os.environ.get("NAME", "World")
    return f"Hi There! {name}"

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        files = request.files.getlist('file')
        if files is None or len(files) < 11:
            return jsonify({'error': 'not enough file'})

        try:
            input_image_url = files[-1]
            verification_urls = files[:-1]
            pred = predict(input_image_url, verification_urls)
            result = ""
            if pred:
                result = 'verified'
            else:
                result = 'unknown'

            return jsonify({'pred': result})

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get("PORT",8081)))
