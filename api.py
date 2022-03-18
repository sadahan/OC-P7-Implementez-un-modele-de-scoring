# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, make_response
import joblib
import pandas as pd, numpy as np
# from lime.lime_tabular import LimeTabularExplainer
# import matplotlib.pyplot as plt
# import mpld3

app = Flask(__name__)

@app.route("/")
def run():
    return "The API is running!"


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        model = joblib.load("model/model.jbl")
        sample = pd.read_csv('data/sample.csv', index_col=0)
    except:
        result = "Files couldn't be loaded."
        return make_response(result, 400)
    
    try:
        id_customer = int(request.args.get('id_customer'))
        customer = np.array(sample.loc[id_customer]).reshape(1, -1)
    except:
        result = 'Please enter a valid customer id.'
        return make_response(result, 400)

    proba = model.predict_proba(customer)
    pred = int(model.predict(customer)[0])
    proba_0 = str(round(proba[0][0]*100,1)) + '%'
    proba_1 = str(round(proba[0][1]*100,1)) + '%'
    result = {'pred': pred,
              'proba_0': proba_0,
              'proba_1': proba_1}
    return jsonify(result)


# @app.route('/lime', methods=['GET', 'POST'])
# def lime_explainer():
#     try:
#         model = joblib.load("model/model.jbl")
#         sample = pd.read_csv('data/sample.csv', index_col=0)
#     except:
#         result = "Files couldn't be loaded."
#         return make_response(result, 400)

#     try:
#         id_customer = int(request.args.get('id_customer'))
#         customer = sample.loc[id_customer]
#     except:
#         result = 'Please enter a valid customer id.'
#         return make_response(result, 400)
    
#     lime = LimeTabularExplainer(sample,
#                                 feature_names=sample.columns,
#                                 class_names=["Solvent", "Not Solvent"],
#                                 discretize_continuous=False)
#     exp = lime.explain_instance(customer,
#                                 model.predict_proba,
#                                 num_samples=100)

#     # exp.show_in_notebook(show_table=True)
#     fig = exp.as_pyplot_figure()
#     plt.tight_layout()

#     return jsonify(fig.as_html())
#     # return jsonify(mpld3.fig_to_html(fig))


if __name__ == "__main__":
    app.run(debug=True)