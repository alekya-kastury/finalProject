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
query="""SELECT 
    C.c_customer_sk,
    CASE 
        WHEN MAX(D_YEAR)=2002 THEN 'Active'
        WHEN MAX(D_YEAR)=2001 THEN 'Inactive'
        WHEN MAX(D_YEAR)<2001 THEN 'Lost'
    END AS customer_status
from
CUSTOMER C INNER JOIN STORE_SALES SS 
ON C.C_CUSTOMER_SK=SS.SS_CUSTOMER_SK
INNER JOIN DATE_DIM DD 
ON C.C_LAST_REVIEW_DATE=DD.D_DATE_SK 
group by c_customer_sk
limit 10;"""

query="""select count(*) from 
STORE_SALES C INNER JOIN DATE_DIM DD ON C.SS_SOLD_DATE_SK=DD.D_DATE_SK
where D_YEAR =2002;"""

@st.cache_data
def run_query(query):
    df=pd.read_sql_query(query,engine)
    return df

st.write(run_query(query))
