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

