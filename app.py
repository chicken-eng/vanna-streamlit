import streamlit as st
import requests
import json

API_URL = "https://app.vanna.ai/api/v0/chat_sse"

headers = {
    "Content-Type": "application/json",
    "VANNA-API-KEY": st.secrets["VANNA_API_KEY"]
}
data = {
    "message": "how many customers are there?",
    "agent_id": "chinook-11",
    "acceptable_responses": ["text", "sql", "dataframe", "plotly"]
}

response = requests.post(API_URL, headers=headers, data=json.dumps(data), stream=True)
st.write("Status code:", response.status_code)

for line in response.iter_lines():
    if line:
        decoded = line.decode('utf-8')
        st.write(decoded)  # show everything raw
