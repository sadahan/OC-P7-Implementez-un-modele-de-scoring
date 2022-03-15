import pandas as pd, numpy as np
import streamlit as st
import streamlit.components.v1 as components
import requests
import joblib
from lime.lime_tabular import LimeTabularExplainer
# import matplotlib.pyplot as plt
import base64


def request_prediction(id_customer):
    API_URL = 'http://127.0.0.1:5000/predict'

    data_json = {'id_customer': id_customer}
    response = requests.get(API_URL, params=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


# def request_explanation(id_customer):
#     API_URL = 'http://127.0.0.1:5000/lime'

#     data_json = {'id_customer': id_customer}
#     response = requests.get(API_URL, params=data_json)

#     if response.status_code != 200:
#         raise Exception(
#             "Request failed with status {}, {}".format(response.status_code, response.text))

#     return response.json()

def lime_explainer(id_customer):
    try:
        model = joblib.load("lgbm_model.joblib")
        X_test = pd.read_csv('X_test.csv', index_col=0)
    except:
        st.error("Files couldn't be loaded.")
    try:
        customer = X_test.loc[id_customer]
    except:
        st.error('Please enter a valid customer id.')
    
    lime = LimeTabularExplainer(X_test,
                                feature_names=X_test.columns,
                                class_names=["Solvent", "Not Solvent"],
                                discretize_continuous=False)
    exp = lime.explain_instance(customer,
                                model.predict_proba,
                                num_samples=100)

    # exp.show_in_notebook(show_table=True)
    fig = exp.as_pyplot_figure()
    # # plt.tight_layout()
    st.pyplot(fig)
    
    return exp.as_html()

    

def main():
    sample = pd.read_csv('sample.csv', index_col=0)
    customers = sample.index.tolist() + [0]

    st.title('Credit scoring prediction:')

    c1, c2 = st.columns([1, 1])
    id_customer = c1.selectbox('Please select a customer id:',
                               sorted(customers, key=int))

    input_customer = c2.number_input('Please enter a customer id:',
                                     min_value=0, value=0, step=1)

    predict_btn = st.button('Score')

    if input_customer:
        id_customer = input_customer

    lime_btn = None

    if predict_btn:
        pred = request_prediction(id_customer)
        # st.write(pred)
        if pred['pred'] == 0:
            message = 'The customer is reliable ' + \
                      'wih a probability of ' + pred['proba_0']
            st.success(message)
        else:
            message = 'The customer is not reliable ' + \
                      'wih a probability of ' + pred['proba_1']
            st.error(message)
    
    lime_btn = st.button('Individual explanations')

    if lime_btn:
        explainer = lime_explainer(id_customer)
        components.html(explainer, height=800)
        # components.html("<html><body><h1>NO</h1></body></html>", width=200, height=200)


if __name__ == '__main__':
    main()
