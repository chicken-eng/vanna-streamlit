import streamlit as st
from vanna.remote import VannaDefault
import vanna

st.write("Vanna version:", vanna.__version__)

api_key = st.secrets.get("VANNA_API_KEY")
vn = VannaDefault(api_key=api_key, model='chinook-11')

# Try alternate method
try:
    st.write("Training data:", vn.get_training_data())
except Exception as e:
    st.write("get_training_data error:", str(e))

# Try a direct SQL generation to see if that works independently
try:
    sql = vn.generate_sql("how many customers are there?")
    st.write("SQL generated:", sql)
except Exception as e:
    st.write("generate_sql error:", str(e))
