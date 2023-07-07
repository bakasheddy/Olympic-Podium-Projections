import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


import pickle

df_sample = pd.read_csv('sample_data.csv')
df = pd.read_csv('modeling_set.csv')
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

#### Sample Table
    """
             )
    st.dataframe(df_sample.head(20))
    st.write("""
        by Shedrack David
        connect with me on [linkedin](https://www.linkedin.com/in/shedrack-david-1a116b235/)
        
        click [here](https://github.com/bakasheddy/ConCop.git) to view project on github, and check out my [Portfolio](bakasheddy.github.io/Portfolio/)
        """)


elif nav == 'Predictions':
    st.image('./image.JPG')
    st.sidebar.subheader('set parameters for predictions')

    def user_input_features():

        gender = st.sidebar.selectbox('Sex', ['male', 'female'], index=1)
        gender = 1 if gender.lower() == 'male' else 0

        age = st.sidebar.slider('set age', value=20,
                                max_value=80, min_value=12)

        NOC = st.sidebar.selectbox(
            'NOC(Country)', ['USA', 'RUS', 'GER', 'CHN', 'CAN', 'GBR', 'AUS', 'FRA', 'KOR', 'NED'], index=1)
        if NOC == 'USA':
            NOC = 0
        elif NOC == 'RUS':
            NOC = 1
        elif NOC == 'GER':
            NOC = 2
        elif NOC == 'CHN':
            NOC = 3
        elif NOC == 'CAN':
            NOC = 4
        elif NOC == 'GBR':
            NOC = 5
        elif NOC == 'AUS':
            NOC = 6
        elif NOC == 'FRA':
            NOC = 7
        elif NOC == 'KOR':
            NOC = 8
        elif NOC == 'NED':
            NOC = 9

        Height = st.sidebar.number_input(
            'Height', max_value=226.0, min_value=133.0)
        Weight = st.sidebar.number_input(
            'Weight', max_value=214.0, min_value=28.0)

        decoded_labels = label_encoder.inverse_transform(df['sport_en'])
        Sport = st.sidebar.selectbox(
            'Select Sport', decoded_labels.values, index=1)

        data = {
            'sport': sport_en,
            'NOC': NOC,
            'gender': Sex,
            'Height': Height,
            'Weight': Weight,
        }
        feautres = pd.DataFrame(data, index=[0])
        return feautres
    dff = user_input_features()
    st.header('Specified Parameters')
    st.write(dff)
    st.write('---')
    file_name = 'olympics.pkl'
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
