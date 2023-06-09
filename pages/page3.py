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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Churn Forecast", page_icon=":bar_chart:", layout="wide")

st.title("Customer Churn Forecast")
#########################################################################################################
#trying chemy
engine = create_engine(URL(
    user= st.secrets["user"],
    password= st.secrets["password"],
    account= st.secrets["account"],
    warehouse= st.secrets["warehouse"],
    database= st.secrets["database"],
    schema= st.secrets["schema"]
))

#####################################################################################################

# Define a function to be executed in parallel
@st.cache_data
def execute_query(query):
    df = pd.read_sql_query(query, engine)
    return df


######################################################################################################
# Define your SQL queries
query1 =  """SELECT * FROM CUSTOMER_DEMO_VIEW;"""
@st.cache_data
def exec_cust_demo(query):
    df_customer_demo=pd.read_sql_query(query, engine)
    return df_customer_demo
df_customer_demo=exec_cust_demo(query1)

query2="""SELECT * FROM CUSTOMER_INCOME;"""
@st.cache_data
def exec_cust_income(query):
    df_customer_income=pd.read_sql_query(query, engine)
    return df_customer_income
df_customer_income=exec_cust_income(query2)

query3= """SELECT * FROM INCOME_VIEW;"""
@st.cache_data
def exec_cust_income_view(query):
    df_income_view=pd.read_sql_query(query, engine)
    return df_income_view
df_income_view=exec_cust_income_view(query3)
#########################################################################################
#### Data Preparation
df_customer_demo=df_customer_demo.dropna()
#########################################################################################
###### Label encoding


# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder()

df_customer_demo['cd_gender']= label_encoder.fit_transform(df_customer_demo['cd_gender'])
df_customer_demo['cd_education_status']= label_encoder.fit_transform(df_customer_demo['cd_education_status'])
df_customer_demo['cd_credit_rating']= label_encoder.fit_transform(df_customer_demo['cd_credit_rating'])
df_customer_demo['cd_marital_status']= label_encoder.fit_transform(df_customer_demo['cd_marital_status'])

###############################################################################3

X = df_customer_demo.drop(columns=['c_customer_sk','c_first_name','c_last_name','customer_status_i'], axis = 1)
y = df_customer_demo['customer_status_i']

from imblearn.over_sampling import SMOTE
smote = SMOTE()

@st.cache_data
def run_model():
    XX_resampled, y_resampled = smote.fit_resample(X,y)
    XX_train, XX_test, y_train, y_test = train_test_split(XX_resampled, y_resampled, test_size = 0.2, random_state = 42)
    random = RandomForestClassifier(n_estimators = 200, max_depth=200, random_state = 0) 
    random.fit(XX_train , y_train) 
    y_pred=random.predict(XX_test)
    XX_test['customer_status_i']=y_pred
    return XX_test

customer_demo_df=run_model()

# replace 'Male' with 1 and 'Female' with 0 in the 'Gender' column
customer_demo_df['cd_gender'] = customer_demo_df['cd_gender'].replace({1:'Male',0:'Female'})
age_bins = [0, 30, 50,100]
age_labels = ['0','30', '50']
customer_demo_df['age_agg'] = pd.cut(customer_demo_df['age'], bins=age_bins, labels=age_labels, include_lowest=True)


customer_demo_df['Segment'] = customer_demo_df['age_agg'].astype(str) + '_' + customer_demo_df['cd_gender'].astype(str)

# create labels for the bins
segment_labels = {'0_Male':'Boy','0_Female':'Girl','30_Male':'Young Adult Male', '30_Female':'Young Adult Female', '50_Male':'Old Male', '50_Female':'Old Female'}

customer_demo_df['Segmented']=customer_demo_df['Segment'].map(segment_labels)

# segment customers based on the combined column
#customer_demo_df['Segmented'] = pd.cut(customer_demo_df['Segment'], bins=segment_bins, labels=segment_labels)

risky_customers=customer_demo_df[customer_demo_df['customer_status_i']==1].shape[0]
retention_rate=round(customer_demo_df[customer_demo_df['customer_status_i']==2].shape[0]*100/customer_demo_df['customer_status_i'].shape[0],2)
###############################################################################
query4=""" SELECT CUSTOMER_STATUS,COUNT(C_CUSTOMER_SK) AS COUNT_OF_CUSTOMERS FROM ACTIVE_CUSTOMERS GROUP BY CUSTOMER_STATUS;"""

@st.cache_data
def exec_status(query):
    df_status=pd.read_sql_query(query, engine)
    return df_status
df_status=exec_status(query4)
################################# CUSTOMER INCOME #################################################3

query5="""SELECT * FROM CUSTOMER_INCOME;"""
@st.cache_data
def exec_cust_inc(query):
    df_customer_income=pd.read_sql_query(query, engine)
    return df_customer_income

df_customer_income=exec_cust_inc(query5)

@st.cache_data
def run_model1():
    X = df_customer_income.drop(columns=['c_customer_sk','customer_status_i'], axis = 1)
    y = df_customer_income['customer_status_i']

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X,y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    random = RandomForestClassifier(n_estimators = 200, max_depth=100, random_state = 0) 
    random.fit(X_train , y_train) 

    y_pred=random.predict(X_test)

    X_test['customer_status_i']=y_pred
    return X_test

X_test=run_model1()
# filter the DataFrame based on a condition
filtered_df = X_test.loc[X_test['customer_status_i'] == 0]

# calculate the mean of column 'B' in the filtered DataFrame
mean_b = filtered_df['income'].mean()
#############################################################
# Create a container for the metrics
with st.beta_container():
    # Create two columns for the metrics
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.metric(label="Risky Customers", value=risky_customers)
    with col2:
        st.metric('Income of Risky Customers', mean_b)
    with col3:
        st.metric('Retention Rate', str(retention_rate)+'%')
############################################ PRODUCT ANALYSIS #######################################

query6= """SELECT * FROM PRODUCT_VIEW;"""

@st.cache_data
def exec_product(query):
    df_product_view=pd.read_sql_query(query, engine)
    return df_product_view
df_product_view =exec_product(query6)
df_product_view=df_product_view.dropna()
@st.cache_data
def run_model2():
    count_threshold = 1000
    df_filtered = df_product_view[df_product_view['i_class'].map(df_product_view['i_class'].value_counts()) >= count_threshold]
# Filter the top 10 products based on their frequency
#top_10_products = df_filtered['i_item_id'].value_counts().nlargest(10)

    df_filtered['cd_gender']= label_encoder.fit_transform(df_filtered['cd_gender'])
    df_filtered['cd_credit_rating']= label_encoder.fit_transform(df_filtered['cd_credit_rating'])
    df_filtered['cd_marital_status']= label_encoder.fit_transform(df_filtered['cd_marital_status'])
    df_filtered['cd_education_status']= label_encoder.fit_transform(df_filtered['cd_education_status'])
    df_filtered['i_class']= label_encoder.fit_transform(df_filtered['i_class'])
    df_product=df_filtered[['age','cd_gender','cd_education_status','cd_credit_rating','cd_marital_status','cd_purchase_estimate','i_class']]

    X = df_product.drop(columns=['i_class'], axis = 1)
    y = df_product['i_class']
    X_resampled, y_resampled = smote.fit_resample(X,y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.2, random_state = 42)
    random = RandomForestClassifier(n_estimators = 200, max_depth=100, random_state = 0) 
    random.fit(X_train , y_train) 
    y_pred=random.predict(X_test)

    X_test['i_class']=y_pred
    return X_test

cust_product_df=run_model2()

#cust_product_df['i_class'].unique()

i_class={0:'accessories',
1:'athletic',
2:'classical',
3:'country',
4:'dresses',
5:'fragrances',
6:'infants',
7:'kids',
8:'maternity',
9:'mens',
10:'newborn',
11:'pants',
12:'pop',
13:'rock',
14:'school-uniforms',
15:'shirts',
16:'sports-apparel',
17:'swimwear',
18:'toddlers',
19:'womens'}

cust_product_df['category']=cust_product_df['i_class'].map(i_class)
cust_product_df['cd_gender'] = cust_product_df['cd_gender'].replace({1:'Male', 0:'Female'})
cust_product_df['age'] = cust_product_df['age'].astype(int)
#cust_product_df[['cd_gender','age','category']]

# define the scoring function
def segment_score(segment_df, category):
    # count the number of purchases in the category for each segment
    segment_counts = segment_df[segment_df['category'] == category].groupby(['cd_gender', 'age']).size().reset_index(name='count')
    # normalize the counts to get the likelihood of purchase for each segment
    segment_counts['score'] = segment_counts['count'] / segment_counts['count'].sum()
    # merge the likelihood scores with the original DataFrame
    segment_scores = segment_df.merge(segment_counts[['cd_gender', 'age', 'score']], on=['cd_gender', 'age'])
    return segment_scores

# calculate the scores for all categories and concatenate the results
scored_df = pd.concat([segment_score(cust_product_df, category) for category in cust_product_df['category'].unique()])

############################################## Dashboard #############################################3


c1 = alt.Chart(df_status,title='Customers by status').mark_bar().encode(x='customer_status', y='count_of_customers')
c1 = c1.properties(width=800, height=400)

# group the DataFrame by the 'Segmented' and 'Status' columns and count the number of customers in each group
segment_status_counts = customer_demo_df.groupby(['Segmented', 'customer_status_i']).size().reset_index(name='Count')

# pivot the DataFrame to create a stacked bar chart
segment_status_pivot = segment_status_counts.pivot(index='customer_status_i', columns='Segmented', values='Count')




tab1, tab2,tab3= st.tabs(["Customers by Status","Product Category Score","Segmentwise Status"])

with tab1:
    st.altair_chart(c1)

with tab2:
    st.write('Product Category Score for each :')
    st.write(scored_df[['cd_gender','age','category','score']])
with tab3:
    st.write ("The Status 0 represents Active customer, Status 1 represents inactive customers and Status 2 represents Lost customers")
    st.write(segment_status_pivot)
    
