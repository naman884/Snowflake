import os
import sys
import joblib
import cachetools
import pandas as pd

from dotenv import load_dotenv

from snowflake.snowpark import Session
from snowflake.snowpark.functions import udf
from snowflake.snowpark import functions as F

#---------------------------------------------------------------------------------------------#

# Establishing connection to snowflake using snowpark
def initiateSession(): 
    
    load_dotenv()
    connection_parameters = {
                                "account": os.getenv('account_name'),
                                "user": os.getenv('user_name'),
                                "password": os.getenv('password'),
                                "role": os.getenv('role'), 
                                "warehouse": os.getenv('warehouse'),
                                "database": os.getenv('database'),
                                "schema": os.getenv('schema')
                            }
    
    session = Session.builder.configs(connection_parameters).create()
    return session

session = initiateSession()

#---------------------------------------------------------------------------------------------#

session.add_packages('snowflake-snowpark-python', 'scikit-learn',
                     'pandas', 'numpy', 'joblib', 'cachetools')

#---------------------------------------------------------------------------------------------#

# Create UDF for Prediction Serving
session.add_import("@house_model_output_stg/housing_price_reg.joblib") 

#---------------------------------------------------------------------------------------------#

'''
Lets say we need to call prediction UDF every 10 mins or evry hr,
then with thehelp of "@cachetools" we will avoid repeating loading of the
model artifacts (in this case its .joblib file) from our stage into the UDF.
Avoid using "@cachetools" only when you have brand new model file (which is new joblib file)
'''

# this func will read the joblib file
@cachetools.cached(cache={})
def read_file(filename):
       import_dir = sys._xoptions.get("snowflake_import_directory")
       if import_dir:
              with open(os.path.join(import_dir, filename), 'rb') as file:
                     m = joblib.load(file)
                     return m

#---------------------------------------------------------------------------------------------#

features = ['LONGITUDE', 'LATITUDE', 'HOUSING_MEDIAN_AGE', 'TOTAL_ROOMS',
            'TOTAL_BEDROOMS', 'POPULATION', 'HOUSEHOLDS', 'MEDIAN_INCOME', 'OCEAN_PROXIMITY']


@udf(name="predict_house_value", is_permanent=True,
stage_location="@house_model_serving_udf_stg", replace=True)
def predict_house_value(LONGITUDE: float,
                        LATITUDE: float, 
                        HOUSING_MEDIAN_AGE: float,
                        TOTAL_ROOMS: float, 
                        TOTAL_BEDROOMS: float, 
                        POPULATION: float, 
                        HOUSEHOLDS: float, 
                        MEDIAN_INCOME: float, 
                        OCEAN_PROXIMITY: str) -> float:
       
       m = read_file('housing_price_reg.joblib')       
       row = pd.DataFrame([locals()], columns=features)
       return m.predict(row)[0]

#---------------------------------------------------------------------------------------------#

# Run Predictions: Way-1

snowdf_test = session.table("HOUSING_TEST")
inputs = snowdf_test.drop("MEDIAN_HOUSE_VALUE")

snowdf_results = snowdf_test.select(predict_house_value(*inputs).alias('predicted_value'), 
                                    (F.col('MEDIAN_HOUSE_VALUE')).alias('actual_value')).limit(20)

snowdf_results = snowdf_results.to_pandas()
print(snowdf_results)


# Run Predictions: Way-2 (from project's point of view this is more useful)
query = ''' select predict_house_value(-122.26,37.85,50.0,1120.0,283.0,697.0,264.0,2.125,'NEAR BAY') '''
result = session.sql(query).collect()
print(result)

