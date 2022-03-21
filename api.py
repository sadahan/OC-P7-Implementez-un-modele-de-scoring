# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, make_response
import joblib
import pandas as pd, numpy as np
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
# import sys
import json

app = Flask(__name__)

@app.route("/")
def run():
    return "The API is running!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        model = joblib.load("model/model.jbl")
        sample = pd.read_csv('data/sample.csv', index_col=0)
        scaler = joblib.load("model/scaler.jbl")
    except:
        result = "Files couldn't be loaded."
        return make_response(result, 400)
    
    try:
        data_customer = request.args.get('data_customer')
        data = pd.DataFrame(json.loads(data_customer), index=[0])
        num_cols = [col for col in sample if not np.isin(sample[col].unique(), [0, 1]).all()]
        # print(len(num_cols), file=sys.stderr)
        data[num_cols] = scaler.transform(data[num_cols].values)
        data = data.iloc[0]
        customer = np.array(data).reshape(1, -1)
    except:
        result = 'Please enter a valid customer id.'
        return make_response(result, 400)

    try:
        proba = model.predict_proba(customer)
        pred = int(model.predict(customer)[0])
        proba_0 = str(round(proba[0][0]*100,1)) + '%'
        proba_1 = str(round(proba[0][1]*100,1)) + '%'
        result = {'pred': pred,
                'proba_0': proba_0,
                'proba_1': proba_1}
        return jsonify(result)
    except:
        return jsonify(data_customer)


@app.route('/lime', methods=['GET', 'POST'])
def lime_explainer():
    try:
        model = joblib.load("model/model.jbl")
        sample = pd.read_csv('data/sample.csv', index_col=0)
        scaler = joblib.load("model/scaler.jbl")
    except:
        result = "Files couldn't be loaded."
        return make_response(result, 400)

    try:
        data_customer = request.args.get('data_customer')
        data = pd.DataFrame(json.loads(data_customer), index=[0])
        num_cols = [col for col in sample if not np.isin(sample[col].unique(), [0, 1]).all()]
        data[num_cols] = scaler.transform(data[num_cols].values)
        sample[num_cols] = scaler.transform(sample[num_cols].values)
        customer = data.iloc[0]
    except:
        result = 'Please enter a valid customer id.'
        return make_response(result, 400)
    
    lime = LimeTabularExplainer(sample,
                                feature_names=sample.columns,
                                class_names=["Solvent", "Not Solvent"],
                                discretize_continuous=False)
    exp = lime.explain_instance(customer,
                                model.predict_proba,
                                num_samples=100)

    return jsonify(exp.as_html())


if __name__ == "__main__":
    app.run(debug=True)