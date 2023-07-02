import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pickle

df = pd.read_csv('sample_data.csv')
nav = st.sidebar.radio("Navigations", ['Home', 'Predictions'])

if nav == "Home":
    st.write(
        """
    # ConCop
    """)

    st.image('./images/Payment-Fraud-Detection_Overgraph.jpg')

    st.write("""### About dataset

The lack of legitimate datasets on mobile money transac-tions to perform research on in the domain of fraud detection is a big problem today in the scientific community. Part of the problem is the intrinsic private nature of financial transactions, that leads to no public available datasets. This will leave the researchers with the burden of first harnessing the dataset before performing the actual researchon it. This paper propose an approach to such a problemthat we named the PaySim simulator.

PaySim is a financial simulator that simulates mobilemoney transactions based on an original dataset. In thispaper, we present a solution to ultimately yield the pos-sibility to simulate mobile money transactions in such away that they become similar to the original dataset. Withtechnology frameworks such as Agent-Based simulationtechniques, and the application of mathematical statistics,we show in this paper that the simulated data can be asprudent as the original dataset for research.
This particular dataset was gotten from kaggle, it contains 6,362,620 data points with 11 columns which captures transactions that has occured in the simulator, both legitimate and fraudulent and this data is what the model was trained on.

Below are the column reference:

- step: represents a unit of time where 1 step equals 1 hour
- type: type of online transaction
- amount: the amount of the transaction
- nameOrig: customer starting the transaction
- oldbalanceOrg: balance before the transaction
- newbalanceOrig: balance after the transaction
- nameDest: recipient of the transaction
- oldbalanceDest: initial balance of recipient before the transaction
- newbalanceDest: the new balance of recipient after the transaction
- isFraud: fraud transaction

link to paper [here](https://www.researchgate.net/publication/313138956_PAYSIM_A_FINANCIAL_MOBILE_MONEY_SIMULATOR_FOR_FRAUD_DETECTION)
    """
             )
    st.dataframe(df.head(20))
    st.write("""
        by Shedrack David
        connect with me on [linkedin](https://www.linkedin.com/in/shedrack-david-1a116b235/)
        
        click [here](https://github.com/bakasheddy/ConCop.git) to view project on github, and check out my [Portfolio](bakasheddy.github.io/Portfolio/)
        """)


elif nav == 'Predictions':
    st.image('./images/Payment-Fraud-Detection_Overgraph.jpg')
    st.sidebar.subheader('set parameters for predictions')

    def user_input_features():

        step = st.sidebar.number_input('step', min_value=1, max_value=743)

        type = st.sidebar.selectbox(
            'type', ['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'], index=1)
        if type == 'CASH_OUT':
            type = 1
        elif type == 'PAYMENT':
            type = 2
        elif type == 'CASH_IN':
            type = 3
        elif type == 'TRANSFER':
            type = 4
        elif type == 'DEBIT':
            type = 5

        amount = st.sidebar.number_input('amount', max_value=9.244552e+07)
        oldbalanceOrg = st.sidebar.number_input(
            'oldbalanceOrg', max_value=5.958504e+07)
        newbalanceOrig = st.sidebar.number_input(
            'newbalanceOrig', max_value=4.958504e+07)
        oldbalanceDest = st.sidebar.number_input(
            'oldbalanceDest', max_value=3.560159e+08)
        newbalanceDest = st.sidebar.number_input(
            'newbalanceDest', max_value=3.561793e+08)

        data = {
            'step': step,
            'type': type,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest
        }
        feautres = pd.DataFrame(data, index=[0])
        return feautres
    dff = user_input_features()
    st.header('Specified Parameters')
    st.write(dff)
    st.write('---')
    file_name = 'ConCop_model.pkl'
    loaded_model = pickle.load(open(file_name, 'rb'))
    predictions = loaded_model.predict(dff)

    st.write('fraud probability (%)')
    prob = loaded_model.predict_proba(dff) * 100
    st.write(prob, '1 means yes, 0 means no')
    st.write('---')

    st.write('is this transaction fraudulent?')
    if predictions == 1:
        predictions = 'yes'
    else:
        predictions = 'no'
    st.write(predictions)
    st.write('---')

    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(
        loaded_model.feature_importances_, index=dff.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Most important features for prediction')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    plt.style.use('ggplot')
    plt.grid(visible=False)
    st.pyplot()
