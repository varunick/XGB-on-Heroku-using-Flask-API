from flask import Flask, jsonify,  request, render_template
from sklearn.externals import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model_load = joblib.load("./models/xgb_model.pkl")
cols_when_model_builds = model_load.get_booster().feature_names

@app.route('/')
def home():
    return render_template('index_xgb.html')

@app.route("/predict", methods=['POST'])
def predict():
    if (request.method == 'POST'):
        int_features = [x for x in request.form.values()]
        final_features_df = pd.DataFrame([int_features],columns = cols_when_model_builds)
        cols = final_features_df.columns
        final_features_df[cols] = final_features_df[cols].apply(pd.to_numeric, errors='coerce')
        output = list(model_load.predict(final_features_df))
        return render_template('index_xgb.html', prediction_text='Interest Output {}'.format(output))
    else :
        return render_template('index_xgb.html')

@app.route("/predict_api", methods=['POST', 'GET'])
def predict_api():
    print(" request.method :",request.method)
    if (request.method == 'POST'):
        data = request.get_json()
        return jsonify(model_load.predict([np.array(list(data.values()))]).tolist())
    else:
        return render_template('index_xgb.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)