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
#distinct_year_query = """SELECT DISTINCT DD.D_YEAR FROM DATE_DIM DD WHERE DD.D_YEAR IN (1998,1999,2000,2001,2002);"""
#distinct_year = pd.read_sql_query(distinct_year_query, engine)['d_year'].tolist()

# create a dropdown for the year parameter with the distinct state values
year = st.sidebar.selectbox('Year', [1998,1999,2000,2001,2002])

# get the distinct year from the database
#distinct_month_query = """SELECT DISTINCT DD.D_MOY FROM DATE_DIM DD;"""
#distinct_month = pd.read_sql_query(distinct_month_query, engine)['d_moy'].tolist()

# create a dropdown for the year parameter with the distinct state values
month = st.sidebar.selectbox('Month', [1,2,3,4,5,6,7,8,9,10,11,12])


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
#st.write (df)

val=df['sales']

def shorten_num(number):
    suffixes = {"K": 1000, "M": 1000000, "B": 1000000000}

    for suffix, dividing_factor in suffixes:
        if number >= dividing_factor:
            shortened_num = str(round(number/dividing_factor, 1)) + suffix
            break
    else:
        shortened_num = str(number)
    return shortened_num

# Create a container for the metrics
with st.beta_container():
    # Create two columns for the metrics
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        st.metric(label="Revenue", value=shorten_num(val))
    with col2:
        st.metric('New Customers', '200')
    with col3:
        st.metric('Repeat Purchase Rate', '300')
    with col4:
        st.metric('Average Order Value', '300')
