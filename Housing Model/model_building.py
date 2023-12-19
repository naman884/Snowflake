import os
import io
import joblib
import numpy as np

from dotenv import load_dotenv

import snowflake.snowpark
from snowflake.snowpark import Session
from snowflake.snowpark.functions import sproc

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

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

# Creating internal stages in SF

# In this stage we will upload our Spocs
query = """
        create or replace stage house_model_training_sproc_stg
        directory = (enable = true)
        copy_options = (on_error='skip_file')
        """

print(session.sql(query).collect())


# In this stage we will save our trained model (.joblib file)
query = """
        create or replace stage house_model_output_stg
        copy_options = (on_error='skip_file')
        """

print(session.sql(query).collect())

# In this stage we will upload our model Serving UDF
query = """
        create or replace stage house_model_serving_udf_stg
        directory = (enable = true)
        copy_options = (on_error='skip_file')
        """

print(session.sql(query).collect())

# To see the stages
session.sql("show stages").show()

#---------------------------------------------------------------------------------------------#

# Defining functions for data pre-processing and model training

# To save the trained model
def save_file(session, model, path):
  input_stream = io.BytesIO()
  joblib.dump(model, input_stream)
  session._conn._cursor.upload_stream(input_stream, path)
  return "successfully created file: " + path

# Model building
def train_model(session: snowflake.snowpark.Session) -> float:
    
    snowdf = session.table("HOUSING_DATA")
    snowdf_train, snowdf_test = snowdf.random_split([0.8, 0.2], seed=82)
    
    snowdf_train.write.mode("overwrite").save_as_table("HOUSING_TRAIN")
    snowdf_test.write.mode("overwrite").save_as_table("HOUSING_TEST")
    
    housing_train = snowdf_train.drop("MEDIAN_HOUSE_VALUE").to_pandas() 
    housing_train_labels = snowdf_train.select("MEDIAN_HOUSE_VALUE").to_pandas()
    housing_test = snowdf_test.drop("MEDIAN_HOUSE_VALUE").to_pandas()
    housing_test_labels = snowdf_test.select("MEDIAN_HOUSE_VALUE").to_pandas()


    housing_num = housing_train.drop("OCEAN_PROXIMITY", axis=1)

    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('std_scaler', StandardScaler()),
        ])

    num_attribs = list(housing_num)
    cat_attribs = ["OCEAN_PROXIMITY"]

    preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", OneHotEncoder(), cat_attribs)
        ])

    full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42)),
        ])

    full_pipeline.fit(housing_train, housing_train_labels)

    save_file(session, full_pipeline, "@house_model_output_stg/housing_price_reg.joblib")

    housing_predictions = full_pipeline.predict(housing_test)
    lin_mse = mean_squared_error(housing_test_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    return lin_rmse


# Creating and deploying stored procedure
train_model_sp = sproc(train_model,
                       name = 'train_house_model',
                       stage_location = '@house_model_training_sproc_stg',
                       is_permanent = True,
                       replace = True)


# Invoking the above sproc (this will perform all the steps which we have written inside "train_model" func)
invoke_result = train_model_sp()
print(invoke_result)

# Now just to see the stage content
session.sql("list @house_model_output_stg").show()

# Note: You should see joblib file as a output

