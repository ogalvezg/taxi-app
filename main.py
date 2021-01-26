import streamlit as st
import pandas as pd 

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

# cache will save the function and doesn't goin to run it again
@st.cache
def get_data(filename):
    taxi_data = pd.read_csv(filename)

    return taxi_data


with header:
    st.title('NYC Taxis Data Analysis Project!')
    st.text('In this project we look into NYC taxis transactions in 2019.')

with dataset:
    st.header('NYC Taxi Dataset')
    st.text('Data source: www1.nyc.gov  Taxi & Limousine Commission')

    taxi_data = get_data('data/taxi_data.csv')
    # st.write(taxi_data.head())

    st.subheader('Pick Up Location ID')
    pulocation_dist = pd.DataFrame(taxi_data['PULocationID'].value_counts()).head(50)
    st.bar_chart(pulocation_dist)

with model_training:
    st.header('Model Training')
    st.text('Here you can choose the hyperparameters of the model and see how the performance changes!')

    sel_col, disp_col = st.beta_columns(2)

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10)

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100, 200, 300,'No limit'], index = 0)

    sel_col.text('Here is a list of features in Dataset:')
    sel_col.write(taxi_data.columns)

    input_feature = sel_col.text_input('Which feature should be used as the input feature?', 'PULocationID')

    if n_estimators == 'No limit':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = taxi_data[[input_feature]]    
    y = taxi_data[['trip_distance']]    

    regr.fit(X, y) 
    prediction = regr.predict(y) 

    disp_col.subheader('Mean absolute error:') 
    disp_col.write(mean_absolute_error(y, prediction)) 

    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared score of the model is:') 
    disp_col.write(r2_score(y, prediction)) 
    