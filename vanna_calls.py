import requests
import json
import streamlit as st

API_URL = "https://app.vanna.ai/api/v0/chat_sse"

def call_vanna_api(message: str) -> dict:
    headers = {
        "Content-Type": "application/json",
        "VANNA-API-KEY": st.secrets["VANNA_API_KEY"]
    }
    data = {
        "message": message,
        "agent_id": "chinook-11",
        "acceptable_responses": ["text", "sql", "dataframe", "plotly", "buttons"]
    }
    result = {"text": None, "sql": None, "dataframe": None, "plotly": None}
    
    response = requests.post(API_URL, headers=headers, data=json.dumps(data), stream=True)
    response.raise_for_status()
    
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data:"):
                data_string = decoded_line[5:].strip()
                try:
                    event = json.loads(data_string)
                    t = event.get("type")
                    if t == "text":
                        result["text"] = event.get("text")
                    elif t == "sql":
                        result["sql"] = event.get("query")
                    elif t == "dataframe":
                        result["dataframe"] = event.get("json_table")
                    elif t == "plotly":
                        result["plotly"] = event.get("json_plotly")
                except json.JSONDecodeError:
                    pass
    return result

@st.cache_data(show_spinner="Thinking...")
def generate_questions_cached():
    return ["How many customers are there?", "What are the top selling tracks?", "Show total sales by country"]

@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    result = call_vanna_api(question)
    return result.get("sql")

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    return sql is not None and sql.strip() != ""

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    import pandas as pd
    result = call_vanna_api(sql)
    df_json = result.get("dataframe")
    if df_json:
        return pd.DataFrame(df_json)
    return None

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    return True

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    return None

@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    result = call_vanna_api(f"Generate a chart for: {df.to_string()[:500]}")
    return result.get("plotly")

@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    return []

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    result = call_vanna_api(f"Summarize this data for the question '{question}': {df.to_string()[:500]}")
    return result.get("text")
