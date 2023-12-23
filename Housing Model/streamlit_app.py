'''
Note:
This file's code is meant to run in a Snowsight Editor only.
'''

# Imaport packages
import streamlit as st
from snowflake.snowpark.context import get_active_session

#----------------------------------------------------------------------#

# Get current session
session = get_active_session()

# Set page config
st.set_page_config(layout="wide", initial_sidebar_state='collapsed')

#----------------------------------------------------------------------#

# App layout

col1, col2, col3 = st.columns([3.5,9,0.5])

with col1:
    pass

with col2:
    st.title('Housing Model Inference App')

with col3:
    pass

st.write('---'*30)
st.info('Please fill the below mentioned input parameters inorder to get the prediction.')
st.write('')

col4, col5, col6 = st.columns([1,1,1])


with col4:
    longitude_input = st.number_input('Longitude')
    rooms_input = st.number_input('No. of Rooms')
    households_input = st.number_input('Households')

with col5:
    latitude_input = st.number_input('Latitude')
    bedrooms_input = st.number_input('No. of Bedrooms')
    proximity_list = ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
    ocean_proximity_input = st.selectbox('Ocean Proximity', options=proximity_list)

with col6:
    house_age_input = st.number_input('House Age')
    population_input = st.number_input('Population')
    income_input = st.number_input('Income')


query = f'''select predict_house_value({longitude_input}, {latitude_input},
                                       {house_age_input}, {rooms_input},
                                       {bedrooms_input}, {population_input},
                                       {households_input}, {income_input},
                                       '{ocean_proximity_input}')'''

#----------------------------------------------------------------------#

# Model Inference

st.write('')
col7, col8, col9 = st.columns([2.5,2,1])


with col8:
    predict_flag = False
    if st.button('Predict', key='submit', type='primary'):
        result = session.sql(query).collect()
        price_value = result[0][0]
        predict_flag = True

if predict_flag == True:
    st.write('---'*30)
    st.success(f'Based on the input parameters, the house price is = ${price_value}')

