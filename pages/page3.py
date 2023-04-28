import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt
import snowflake.connector as sf
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
import matplotlib.pyplot as plt
import humanize
import altair as alt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import multiprocessing

st.set_page_config(page_title="Customer Churn Forecast", page_icon=":bar_chart:", layout="wide")

st.title("Customer Churn Forecast")
#########################################################################################################
#trying chemy
engine = create_engine(URL(
    account = 'dl84836.us-east-2.aws',
    user = 'alekyakastury',
    password = '@Noon1240',
    database = 'CUSTOMER',
    schema = 'PUBLIC',
    warehouse = 'compute_wh'
))

#####################################################################################################
query=""" SELECT CUSTOMER_STATUS,COUNT(C_CUSTOMER_SK) AS COUNT_OF_CUSTOMERS FROM ACTIVE_CUSTOMERS GROUP BY CUSTOMER_STATUS;"""

#@st.cache_data

df=pd.read_sql_query(query,engine)

c1 = alt.Chart(df,title='Active customers').mark_bar().encode(x='customer_status', y='count_of_customers')
c1 = c1.properties(width=800, height=400)
st.altair_chart(c1)
######################################################################################################


# Define a function to be executed in parallel
@st.cache_data
def execute_query(query):
    engine = create_engine(URL(
    account = 'dl84836.us-east-2.aws',
    user = 'alekyakastury',
    password = '@Noon1240',
    database = 'CUSTOMER',
    schema = 'PUBLIC',
    warehouse = 'compute_wh'))
    df = pd.read_sql_query(query, engine)
    return df

# Define your SQL queries
queries = [
    """SELECT * FROM CUSTOMER_DEMO_VIEW;""",
    """SELECT * FROM CUSTOMER_INCOME;""",
    """SELECT * FROM INCOME_VIEW;"""
]


# Create a pool of worker processes
pool = multiprocessing.Pool(processes=3)

# Execute the queries in parallel
results = pool.map(execute_query, [(query) for query in queries])

# Unpack the results into separate DataFrames
df_customer_demo, df_customer_income, df_income_view = results

# Output the results
st.write('C1')



