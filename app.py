import string
from tkinter import N
from flask import Flask, redirect, render_template, request, Markup , jsonify
import numpy as np
import pandas as pd
from utils.crop import crop_dic
from utils.disease import disease_dic
import requests
import config
import pickle
import io
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9
from chat import get_response


disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)_------------_Common_rust',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape__Esca(Black_Measles)',
                   'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange__Haunglongbing(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,bell__Bacterial_spot',
                   'Pepper,bell__healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

model = pickle.load(open('models/RFregressor.pkl', 'rb'))
models = pickle.load(open('models/logistic.pkl', 'rb'))


def predict_image(img, model=disease_model):

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    yb = model(img_u)

    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]

    return prediction


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/base')
def base():
    return render_template("base.html")

@app.route('/yeild')
def yeild():
    return render_template('yield.html')

@app.route("/chat", methods=['POST'])
def predictchat():
    text = request.get_json().get("message")
    # TODO: check if text is valid
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

@app.route("/recommendation", methods=["POST"])
def recommendation():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    pred = models.predict(features)
    a = np.array2string(pred)
    my_prediction = Markup(str(crop_dic[a]))
    return render_template("crop-result.html", prediction=my_prediction)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':

        dist = request.form.get('dist')
        season = request.form.get('season')
        crop = request.form.get('crop')
        area = int(request.form['area'])

        data = np.array([[dist, season, crop, area]])
        my_prediction = model.predict(data)
        prod = area*my_prediction

        return render_template('result.html', prediction=my_prediction, production=prod)


@app.route('/disease-predict', methods=['GET', 'POST'])
def disease():
    title = 'Disease Detection'
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


if __name__ == '__main__':
    app.run(debug=True)