from flask import Flask, jsonify,  request, render_template
from sklearn.externals import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model_load = joblib.load("./models/rfc_model.pkl")

@app.route('/')
def home():
    return render_template('index_rfc.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        int_features = [x for x in request.form.values()]
        final_features = [np.array(int_features)]
        output = model_load.predict(final_features).tolist()
        return render_template('index_rfc.html', prediction_text='Interest Output {}'.format(output))
    else :
        return render_template('index_rfc.html')

@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :",request.method)
    if (request.method == 'POST'):
        data = request.get_json()
        return jsonify(model_load.predict([np.array(list(data.values()))]).tolist())
    else:
        return render_template('index_rfc.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)