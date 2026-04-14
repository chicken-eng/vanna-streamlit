import streamlit as st
from vanna.remote import VannaDefault
import vanna

st.write("Vanna version:", vanna.__version__)

api_key = st.secrets.get("VANNA_API_KEY")
vn = VannaDefault(api_key=api_key, model='chinook-11')
st.write("Training data:", vn.get_training_data())
