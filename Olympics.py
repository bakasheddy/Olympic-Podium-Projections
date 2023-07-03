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

    st.image('./image.JPG')

    st.write("""### About dataset

This is a historical dataset on the modern Olympic Games, including all the Games from Athens 1896 to Rio 2016. The data was scraped from www.sports-reference.com in May 2018. 
Below are the column reference:

- ID - Unique number for each athlete
- Name - Athlete's name
- Sex - M or F
- Age - Integer
- Height - In centimeters
- Weight - In kilograms
- Team - Team name
- NOC - National Olympic Committee 3-letter code
- Games - Year and season
- Year - Integer
- Season - Summer or Winter
- City - Host city
- Sport - Sport
- Event - Event
- Medal - Gold, Silver, Bronze, or NA

Note that the Winter and Summer Games were held in the same year up until 1992. After that, they staggered them such that Winter Games occur on a four year cycle starting with 1994, then Summer in 1996, then Winter in 1998, and so on. A common mistake people make when analyzing this data is to assume that the Summer and Winter Games have always been staggered.

### Acknowledgements
The Olympic data on www.sports-reference.com is the result of an incredible amount of research by a group of Olympic history enthusiasts and self-proclaimed 'statistorians'. Check out their [blog](https://olympstats.com/) for more information. All I did was consolidated their decades of work into a convenient format for data analysis.

### Inspiration
This dataset provides an opportunity to ask questions about how the Olympics have evolved over time, including questions about the participation and performance of women, different nations, and different sports and events.
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
