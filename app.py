import streamlit as st
from vanna.remote import VannaDefault

vn = VannaDefault(api_key=st.secrets.get("VANNA_API_KEY"), model='fsi')
st.write(vn.get_training_data())
