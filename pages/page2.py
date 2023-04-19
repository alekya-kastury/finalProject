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

# create a dropdown for the year parameter with the distinct state values
year = st.sidebar.selectbox('Year', [1998,1999,2000,2001,2002])

# create a dropdown for the year parameter with the distinct state values
month = st.sidebar.selectbox('Month', [1,2,3,4,5,6,7,8,9,10,11,12])

#####################################################BLOCK 1##############################################

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

df_rev_current=pd.read_sql_query(query,engine)
revenue_current=df_rev_current['sales'][0]


if year==1998 and month==1:
    percentage=100
elif month==1:
    prev_year=year-1
    prev_month=12
    
    query="""SELECT SUM(SS_NET_PAID) as sales FROM 
    STORE_SALES SS INNER JOIN DATE_DIM DD
    ON SS.SS_SOLD_DATE_SK=DD.D_DATE_SK
    WHERE DD.D_YEAR={} and DD.D_MOY={}
    group by DD.D_YEAR, DD.D_MOY""".format(prev_year,prev_month)
    
else:
    prev_month=month-1
    
    query="""SELECT SUM(SS_NET_PAID) as sales FROM 
    STORE_SALES SS INNER JOIN DATE_DIM DD
    ON SS.SS_SOLD_DATE_SK=DD.D_DATE_SK
    WHERE DD.D_YEAR={} and DD.D_MOY={}
    group by DD.D_YEAR, DD.D_MOY""".format(year,prev_month)
    
df_rev_prev=pd.read_sql_query(query,engine)
revenue_prev=df_rev_prev['sales'][0]

percentage=((revenue_current-revenue_prev)/revenue_prev)*100

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
###########################################################################################################


#################################BLOCK 2##################################################
if year==1998 and month==1:
    no_of_cust=0
elif month==1:
    prev_year=year-1
    prev_month=12
    
    query="""SELECT COUNT(SS_CUSTOMER_SK) as no_of_customers
    FROM STORE_SALES SS INNER JOIN DATE_DIM DD ON
    SS.SS_SOLD_DATE_SK=DD.D_DATE_SK 
    WHERE DD.D_YEAR={} AND DD.D_MOY={}
    AND SS_CUSTOMER_SK NOT IN 
    (
    SELECT DISTINCT SS_CUSTOMER_SK FROM STORE_SALES WHERE SS_SOLD_DATE_SK IN
    (SELECT D_DATE_SK FROM DATE_DIM HAVING D_YEAR BETWEEN (SELECT MIN(D_YEAR) FROM DATE_DIM) AND D_YEAR={})
    );""".format(year,month,prev_year)
    
else:
    prev_month=month-1
    
    query="""SELECT COUNT(SS_CUSTOMER_SK) as no_of_customers
    FROM STORE_SALES SS INNER JOIN DATE_DIM DD ON
    SS.SS_SOLD_DATE_SK=DD.D_DATE_SK 
    WHERE DD.D_YEAR={} AND DD.D_MOY={}
    AND SS_CUSTOMER_SK NOT IN 
    (
    SELECT DISTINCT SS_CUSTOMER_SK FROM STORE_SALES WHERE SS_SOLD_DATE_SK IN
    (SELECT D_DATE_SK FROM DATE_DIM HAVING D_YEAR BETWEEN (SELECT MIN(D_YEAR) FROM DATE_DIM) AND D_YEAR={})
    );""".format(year,month,prev_year)
    
df_no_of_cust=pd.read_sql_query(query,engine)
no_of_customers=df_no_of_cust['no_of_customers'][0]


###########################################################################################


# Create a container for the metrics
with st.beta_container():
    # Create two columns for the metrics
    col1, col2, col3, col4 = st.beta_columns(4)
    with col1:
        st.metric(label="Revenue", value=shorten_num(revenue_current),delta=round(percentage,1))
    with col2:
        st.metric('New Customers', no_of_customers)
    with col3:
        st.metric('Repeat Purchase Rate', '300')
    with col4:
        st.metric('Average Order Value', '300')
