import streamlit as st
from vanna.remote import VannaDefault

api_key = st.secrets.get("VANNA_API_KEY")
st.write("API Key found:", api_key is not None)
# shows first 5 chars only
st.write("API Key value:", api_key[:5] if api_key else "MISSING")

vn = VannaDefault(api_key=api_key, model='chinook-11')
st.write("Vanna object created:", vn is not None)

training_data = vn.get_training_data()
st.write("Training data:", training_data)
st.write("Training data type:", type(training_data))
