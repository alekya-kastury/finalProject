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

#####################################################BLOCK 1##############################################
@st.cache_data
def run_query(query,col):
    df=pd.read_sql_query(query,engine)
    res=df[col][0]
    return res

query="""SELECT SUM(SS_NET_PAID) as sales FROM STORE_SALES SS INNER JOIN DATE_DIM DD ON
SS.SS_SOLD_DATE_SK=DD.D_DATE_SK WHERE 
DD.D_YEAR={} and DD.D_MOY={} group by 
DD.D_YEAR, DD.D_MOY""".format(year,month)

revenue_current=run_query(query,'sales')

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

revenue_prev=run_query(query,'sales')

percentage=((revenue_current-revenue_prev)/revenue_prev)*100


###########################################################################################################


#################################BLOCK 2##################################################
query="""SELECT count(distinct SS_CUSTOMER_SK) as no_of_customers
FROM STORE_SALES SS INNER JOIN DATE_DIM DD ON
SS.SS_SOLD_DATE_SK=DD.D_DATE_SK 
WHERE DD.D_YEAR={} AND DD.D_MOY={}
group by DD.D_YEAR, DD.D_MOY;""".format(year,month)

no_of_customers=run_query(query,'no_of_customers')

if year==1998 and month==1:
    percentage_cust=100
elif month==1:
    prev_year=year-1
    prev_month=12
    query="""SELECT count(distinct SS_CUSTOMER_SK) as no_of_customers
    FROM STORE_SALES SS INNER JOIN DATE_DIM DD ON
    SS.SS_SOLD_DATE_SK=DD.D_DATE_SK 
    WHERE DD.D_YEAR={} AND DD.D_MOY={}
    group by DD.D_YEAR, DD.D_MOY;""".format(prev_year,prev_month)
    
else:
    prev_month=month-1
    query="""SELECT count(distinct SS_CUSTOMER_SK) as no_of_customers
    FROM STORE_SALES SS INNER JOIN DATE_DIM DD ON
    SS.SS_SOLD_DATE_SK=DD.D_DATE_SK 
    WHERE DD.D_YEAR={} AND DD.D_MOY={}
    group by DD.D_YEAR, DD.D_MOY;""".format(year,prev_month)

no_of_customers_prev=run_query(query,'no_of_customers')

percentage_cust=((no_of_customers-no_of_customers_prev)/no_of_customers_prev)*100

###########################################################################################
##############################BLOCK 3####################################################
query="""SELECT count(SS_CUSTOMER_SK) AS ret
FROM (
  SELECT SS_CUSTOMER_SK, COUNT(*) AS count
  FROM STORE_SALES SS INNER JOIN DATE_DIM DD
 ON SS.SS_SOLD_DATE_SK=DD.D_DATE_SK
   WHERE DD.D_YEAR={} and DD.D_MOY={}
  GROUP BY SS_CUSTOMER_SK
) S WHERE count>1 ;""".format(year,month)

ret_customers=run_query(query,'ret')         

query="""SELECT count(SS_CUSTOMER_SK) AS total
FROM (
  SELECT SS_CUSTOMER_SK
  FROM STORE_SALES SS INNER JOIN DATE_DIM DD
 ON SS.SS_SOLD_DATE_SK=DD.D_DATE_SK
   WHERE DD.D_YEAR={} and DD.D_MOY={}
) S ;""".format(year,month)

total_customers=run_query(query,'total')

percentage_ret_customers=ret_customers*100/total_customers
#########################################################################################
#############################BLOCK 4#####################################################

query="""SELECT count(SS_NET_PAID) as count_sales FROM STORE_SALES SS INNER JOIN DATE_DIM DD ON
SS.SS_SOLD_DATE_SK=DD.D_DATE_SK WHERE 
DD.D_YEAR={} and DD.D_MOY={} group by 
DD.D_YEAR, DD.D_MOY;""".format(year,month)

count_sales=run_query(query,'count_sales')
average=revenue_current /count_sales

if year==1998 and month==1:
    percentage_avg_inc=100
elif month==1:
    prev_year=year-1
    prev_month=12
    
    query="""SELECT count(SS_NET_PAID) as prev_count_sales FROM 
    STORE_SALES SS INNER JOIN DATE_DIM DD
    ON SS.SS_SOLD_DATE_SK=DD.D_DATE_SK
    WHERE DD.D_YEAR={} and DD.D_MOY={}
    group by DD.D_YEAR, DD.D_MOY""".format(prev_year,prev_month)
    prev_count=run_query(query,'prev_count_sales')
    prev_avg=revenue_prev/prev_count
    percentage_avg_inc=(average-prev_avg)*100/prev_avg    
else:
    prev_month=month-1
    
    query="""SELECT count(SS_NET_PAID) as prev_count_sales FROM 
    STORE_SALES SS INNER JOIN DATE_DIM DD
    ON SS.SS_SOLD_DATE_SK=DD.D_DATE_SK
    WHERE DD.D_YEAR={} and DD.D_MOY={}
    group by DD.D_YEAR, DD.D_MOY""".format(year,prev_month)

    prev_count=run_query(query,'prev_count_sales')
    prev_avg=revenue_prev/prev_count
    percentage_avg_inc=(average-prev_avg)*100/prev_avg

#########################################################################################
# Create a container for the metrics
with st.beta_container():
    # Create two columns for the metrics
    col1, col2, col3,col4 = st.beta_columns(4)
    with col1:
        st.metric(label="Revenue", value=shorten_num(revenue_current),delta=str(round(percentage,1))+'%')
    with col2:
        st.metric('Number of Customers', shorten_num(no_of_customers),delta=str(round(percentage_cust,1))+'%')
    with col3:
        st.metric('Returning customers', str(round(percentage_ret_customers,1))+'%') 
    with col4:
        st.metric('Average Order Value',round(average,0),delta=str(round(percentage_avg_inc,1))+'%')

#######################################################################################################################
query="""SELECT dd.d_year as YEAR,COUNT(SS_CUSTOMER_SK) AS COUNT_OF_CUSTOMERS
FROM STORE_SALES SS INNER JOIN DATE_DIM DD
ON SS.SS_SOLD_DATE_SK=DD.D_DATE_SK
WHERE DD.D_MOY={}
group by DD.D_YEAR;""".format(month)

@st.cache_data
def run_query_plot(query):
    df=pd.read_sql_query(query,engine)
    st.line_chart(df,axis=0)

run_query_plot(query)
 
