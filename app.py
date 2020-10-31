import numpy as np
import pandas as pd
import sklearn
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [x for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template('index.html', prediction_text='Math marks estimated to be {}'.format(output))



if __name__ == "__main__":
    app.run(debug=True)