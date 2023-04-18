pip-upgrade requirements.txt
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from snowflake.sqlalchemy import URL

st.write ("Welcome")


engine = create_engine(URL(
    account = 'mgvioku-jw32137',
    user = 'alekyakastury',
    password = '@Noon1240',
    database = 'CUSTOMER',
    schema = 'public',
    warehouse = 'COMPUTE_WH',
    role='ACCOUNTADMIN',
))
connection = engine.connect()
try:
    st.write(connection.execute('SELECT * FROM CUSTOMER LIMIT 1'))
finally:
    connection.close()
