import pandas as pd, numpy as np
import streamlit as st
import streamlit.components.v1 as components
import requests
import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from PIL import Image


st.set_option('deprecation.showPyplotGlobalUse', False)


def request_prediction(id_customer):
    API_URL = 'http://127.0.0.1:5000/predict'

    data_json = {'id_customer': id_customer}
    response = requests.get(API_URL, params=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


def lime_explainer(id_customer):
    customer = st.session_state.sample.loc[id_customer]

    lime = LimeTabularExplainer(st.session_state.sample,
                                feature_names=st.session_state.sample.columns,
                                class_names=["Solvent", "Not Solvent"],
                                discretize_continuous=False)
    exp = lime.explain_instance(customer,
                                st.session_state.model.predict_proba,
                                num_samples=100)
    exp.show_in_notebook(show_table=True)
    fig = exp.as_pyplot_figure()
    st.pyplot(fig)
    return exp.as_html()


def shap_explainer(id_customer):
    customer = st.session_state.sample.loc[id_customer]
    i = st.session_state.sample.index.get_loc(id_customer)
    explainer = shap.TreeExplainer(st.session_state.model)
    shap_values = explainer.shap_values(st.session_state.sample)
    st.pyplot(shap.force_plot(matplotlib=True,
                              base_value=explainer.expected_value[1],
                              shap_values=shap_values[1][i],
                              features=st.session_state.sample.loc[id_customer],
                              feature_names=st.session_state.sample.columns
                             ))
    
 
def display_importances(nb_feats):
    feature_importance_values = st.session_state.model.feature_importances_

    feats = pd.DataFrame({'feature': list(st.session_state.sample.columns),
                          'importance': feature_importance_values})
    cols = feats.groupby("feature").mean().sort_values(by="importance", ascending=False)[:nb_feats].index
    best_feats = feats.loc[feats.feature.isin(cols)]
    fig = plt.figure(figsize=(8, 10))
    if nb_feats > 20:
        fig = plt.figure(figsize=(8, 15))
    sns.barplot(x="importance", y="feature", data=best_feats.sort_values(by="importance", ascending=False))
    plt.tight_layout()
    st.sidebar.pyplot(fig)
 

def main():

    try:
        model = joblib.load("model/model.jbl")
        sample = pd.read_csv('data/sample.csv', index_col=0)
    except:
        st.error("Files couldn't be loaded.")
    
    st.session_state.sample = sample
    st.session_state.model = model

    customers = sample.index.tolist() + [0]
    customers = list(map(str, sorted(customers)))

    st.title('Credit scoring prediction:')
    st.sidebar.title('Global importance of features')

    nb_feats = st.sidebar.number_input('Number of features :',
                            min_value=5, max_value = 40, value=10, step=1)
    
    display_importances(nb_feats)
    # col1, col2 = st.columns(2)

    # with col1:
    #     with st.form(key='form'):
    id_customer = st.selectbox('Please select a customer id:',
                                customers)
    id_customer = int(id_customer)
    # predict_btn = st.form_submit_button('Score')
    predict_btn = st.button('Score')

    if predict_btn:
        pred = request_prediction(id_customer)
        if pred['pred'] == 0:
            st.success('The customer is solvent ' + \
                    'wih a probability of ' + pred['proba_0'])
        else:
            st.error('The customer is not solvent ' + \
                    'wih a probability of ' + pred['proba_1'])
    
        explainer = lime_explainer(id_customer)
        shap_explainer(id_customer)
                # json2html.convert(components.html(explainer, height=800))
        components.html(explainer, height=800)
                # components.html("<html><body><h1>NO</h1></body></html>", width=200, height=200)
    


if __name__ == '__main__':
    main()
