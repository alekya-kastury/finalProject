import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime as dt
import snowflake.connector as sf
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL
import matplotlib.pyplot as plt

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
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.success("Select a demo above.")

st.write ("Welcome")
query="""SELECT * FROM CUSTOMER LIMIT 1;"""
df=pd.read_sql_query(query,engine)
st.write(df)

def shorten_num(number):    
    if number >= 1000000000:
        shortened_num = str(round(number/1000000000, 1)) + "B"
    elif number >= 1000000:
        shortened_num = str(round(number/1000000, 1)) + "M"
    elif number >= 1000:
        shortened_num = str(round(number/1000, 1)) + "K"
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

