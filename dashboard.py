import pandas as pd, numpy as np
import streamlit as st
import streamlit.components.v1 as components
import requests
import joblib
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from lime.lime_tabular import LimeTabularExplainer
import json
from sklearn.neighbors import NearestNeighbors


st.set_option('deprecation.showPyplotGlobalUse', False)


def request_prediction(data_customer):
    API_URL = 'http://127.0.0.1:5000/predict'

    data_json = {'data_customer': data_customer}
    response = requests.get(API_URL, params=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    return response.json()

# def lime_explainer(data_customer):
#     data = pd.DataFrame(json.loads(data_customer), index=[0])
#     data[st.session_state.num_cols] = st.session_state.scaler.transform(data[st.session_state.num_cols].values)
#     customer = data.iloc[0]

#     lime = LimeTabularExplainer(st.session_state.sample,
#                                 feature_names=st.session_state.sample.columns,
#                                 class_names=["Solvent", "Not Solvent"],
#                                 discretize_continuous=False)
#     exp = lime.explain_instance(customer,
#                                 st.session_state.model.predict_proba,
#                                 num_samples=100)
#     # exp.show_in_notebook(show_table=True)
#     fig = exp.as_pyplot_figure()
#     st.pyplot(fig)
#     return exp.as_html()

def lime_explainer(data_customer):
    API_URL = 'http://127.0.0.1:5000/lime'

    data_json = {'data_customer': data_customer}
    response = requests.get(API_URL, params=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))
    return response.json()


def shap_explainer(data_customer):
    shap_sample = st.session_state.sample.copy()
    customer = pd.Series(json.loads(data_customer))
    shap_sample = shap_sample.append(customer, ignore_index=True)
    i = shap_sample.shape[0] - 1
    explainer = shap.TreeExplainer(st.session_state.model)
    shap_values = explainer.shap_values(shap_sample)
    st.pyplot(shap.force_plot(matplotlib=True,
                              base_value=explainer.expected_value[1],
                              shap_values=shap_values[1][i],
                              features=customer,
                              feature_names=shap_sample.columns
                             ))
    
def get_closest_neighbors(df, id_customer, n_neighbors=5):
    nn_model = NearestNeighbors(n_neighbors=n_neighbors+1)
    nbrs = nn_model.fit(df)
    list_neighbors = nn_model.kneighbors([df.loc[id_customer]])[1].flatten()
    neighbors_ids = []
    for n in list_neighbors:
        neighbors_ids.append(df.iloc[n].name)
    return neighbors_ids[1:]


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
        scaler = joblib.load("model/scaler.jbl")
    except:
        st.error("Files couldn't be loaded.")

    num_cols = [col for col in sample if not np.isin(sample[col].unique(), [0, 1]).all()]
    unscaled_sample = sample.copy()
    sample[num_cols] = scaler.transform(sample[num_cols].values)

    st.session_state.sample = sample
    st.session_state.num_cols = num_cols
    st.session_state.model = model
    st.session_state.scaler = scaler

    customers = sample.index.tolist()
    customers = list(map(str, sorted(customers)))

    st.title('Credit scoring prediction:')
    st.sidebar.title('Global importance of features')

    nb_feats = st.sidebar.number_input('Number of features :',
                            min_value=5, max_value = 40, value=10, step=1)
    
    display_importances(nb_feats)

    id_customer = st.selectbox('Please select a customer id:',
                                customers)
    id_customer = int(id_customer)

    customer = unscaled_sample.loc[id_customer]

    c1, c2 = st.columns(2)
    c1.number_input('Income of the client (AMT_INCOME_TOTAL) :',
                    min_value=0.0, value=customer['AMT_INCOME_TOTAL'],
                    key='income')
    c2.number_input('Credit amount of the loan (AMT_CREDIT) :',
                    min_value=0.0, value=customer['AMT_CREDIT'],
                    key='credit')
    c1.number_input('Loan annuity (AMT_ANNUITY) :',
                    min_value=0.0, value=customer['AMT_ANNUITY'],
                    key='annuity')
    c2.number_input('Price of the goods for which the loan is given (AMT_GOODS_PRICE) :',
                    min_value=0.0, value=customer['AMT_GOODS_PRICE'],
                    key='goods_price')
    nb_neighbors = c1.number_input('Number of similar clients to show :',
                                    min_value=1, value=5)
    n_neighbor = c2.number_input('Explaining similar clients number :',
                                    min_value=1, value=1, max_value=nb_neighbors)

    customer['AMT_INCOME_TOTAL'] = st.session_state.income
    customer['AMT_CREDIT'] = st.session_state.credit
    customer['AMT_ANNUITY'] = st.session_state.annuity
    customer['AMT_GOODS_PRICE'] = st.session_state.goods_price
    customer['INCOME_CREDIT_PERC'] = customer['AMT_INCOME_TOTAL'] / customer['AMT_CREDIT']
    customer['INCOME_PER_PERSON'] = customer['AMT_INCOME_TOTAL'] / customer['CNT_FAM_MEMBERS']
    customer['ANNUITY_INCOME_PERC'] = customer['AMT_ANNUITY'] / customer['AMT_INCOME_TOTAL']
    customer['PAYMENT_RATE'] = customer['AMT_ANNUITY'] / customer['AMT_CREDIT']

    data_customer = customer.to_json()
    pred = ~model.predict(sample).astype(bool)
    unscaled_sample['solvent'] = pred

    predict_btn = st.button('Score')

    if predict_btn:
        pred = request_prediction(data_customer)
        if pred['pred'] == 0:
            st.success('The customer is solvent wih a probability of ' + pred['proba_0'])
        else:
            st.error('The customer is not solvent wih a probability of ' + pred['proba_1'])
    
        explainer = lime_explainer(data_customer)
        shap_explainer(data_customer)
        components.html(explainer, height=600)

        neighbors = get_closest_neighbors(sample, id_customer, nb_neighbors)
        cols = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                'solvent']
        df_neighbors = unscaled_sample[cols].rename(columns={'AMT_INCOME_TOTAL': 'income',
         'AMT_CREDIT': 'credit', 'AMT_ANNUITY': 'annuity', 'AMT_GOODS_PRICE': 'goods_price'})
        st.write('Features of the ' + str(nb_neighbors) + ' most similar customers:')
        st.write(df_neighbors.loc[neighbors])
        neighbor = sample.loc[neighbors[n_neighbor-1]]
        st.write('Explanation for the prediction of the ' + str(n_neighbor) + 'th most similar customer:')
        explain_neighbor = lime_explainer(neighbor.to_json())
        shap_explainer(neighbor.to_json())
        components.html(explain_neighbor, height=600)

if __name__ == '__main__':
    main()
