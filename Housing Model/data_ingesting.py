import os
import pandas as pd
from dotenv import load_dotenv
from snowflake.snowpark import Session

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

# Creating the table in which we will upload the CSV file data
query = """
        create or replace table HOUSING_DATA (LONGITUDE float,
                                              LATITUDE float,
                                              HOUSING_MEDIAN_AGE float,
                                              TOTAL_ROOMS float,
                                              TOTAL_BEDROOMS float,
                                              POPULATION float,
                                              HOUSEHOLDS float,
                                              MEDIAN_INCOME float,
                                              MEDIAN_HOUSE_VALUE float,
                                              OCEAN_PROXIMITY varchar(255))
                                              """

session.sql('select * from HOUSING_DATA').show()

# Reading local CSV file as a pandas dataframe
df = pd.read_csv('housing.csv')

# Converting df columns to upper case
df.columns = [col.upper() for col in df.columns]

# Now writing the 'df' data into the above created table
session.write_pandas(df, "HOUSING_DATA")

# querying the table (just to see the data)
session.sql('select * from HOUSING_DATA limit 10').show()
