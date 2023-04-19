import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt
import snowflake.connector as sf
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
import matplotlib.pyplot as plt
@st.cache
#trying chemy
engine = create_engine(URL(
    account = 'dl84836.us-east-2.aws',
    user = 'alekyakastury',
    password = '@Noon1240',
    database = 'CUSTOMER',
    schema = 'PUBLIC',
    warehouse = 'compute_wh'
))

st.set_page_config(
    page_title="Customer Analysis Dashboard",
)

# get the distinct year from the database

distinct_year_query = """SELECT DISTINCT DD.D_YEAR FROM 
STORE_SALES SS INNER JOIN DATE_DIM DD
ON SS.SS_SOLD_DATE_SK=DD.D_DATE_SK;"""

distinct_year = pd.read_sql_query(distinct_year_query, engine)['d_year'].tolist()

# create a dropdown for the year parameter with the distinct state values
year = st.sidebar.selectbox('Year', distinct_year)

# get the distinct year from the database

distinct_month_query = """SELECT DISTINCT DD.D_MOY FROM STORE_SALES SS INNER JOIN DATE_DIM DD
ON SS.SS_SOLD_DATE_SK=DD.D_DATE_SK;"""

distinct_month = pd.read_sql_query(distinct_month_query, engine)['d_moy'].tolist()

# create a dropdown for the year parameter with the distinct state values
month = st.sidebar.selectbox('Month', distinct_month)


query="""SELECT SUM(SS_NET_PAID) as sales
FROM 
STORE_SALES SS INNER JOIN DATE_DIM DD
ON
SS.SS_SOLD_DATE_SK=DD.D_DATE_SK
WHERE 
DD.D_YEAR={} and
DD.D_MOY={}
group by 
DD.D_YEAR, DD.D_MOY""".format(year,month)

df=pd.read_sql_query(query,engine)
st.write (df)
#st.metric(label="Total Sales per Month", value=df[sales], delta=None, delta_color="normal", help=None, label_visibility="visible")


