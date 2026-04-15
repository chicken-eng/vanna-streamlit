import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

# ----------------------------
# Database connection
# ----------------------------
@st.cache_resource
def get_engine():
    url = (
        f"postgresql+psycopg2://{st.secrets['DB_USER']}:{st.secrets['DB_PASSWORD']}"
        f"@{st.secrets['DB_HOST']}:{st.secrets.get('DB_PORT', 5432)}/{st.secrets['DB_NAME']}"
        f"?sslmode=require"
    )
    return create_engine(url)

# ----------------------------
# Gemini LLM
# ----------------------------
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0
    )

# ----------------------------
# Schema context
# ----------------------------
SCHEMA_DESCRIPTION = """
You are a data analyst assistant for a market research and panel management company.
The PostgreSQL database contains the following key tables:

- respondent: Core table of panel members. Fields include email (PK), first_name, last_name, 
  date_of_birth, phone_number, gender, ethnicity, is_deleted, ip_address.

- addresses: Respondent address info linked by email. Fields include country, uk_region, 
  county_state, city, postal_code.

- respondent_type_specification: What type a respondent is (consumer, HCP, patient, B2B etc), 
  linked by email and type_id. Includes is_active, respondent_tier.

- respondent_type: Lookup table for type names (type_id, type_name).

- conditions: Medical conditions lookup (condition_id, condition).

- respondent_condition_specification: Links respondents to their conditions via email and 
  condition_id. Includes professionally_diagnosed flag.

- hcp_job_title: Lookup for HCP job titles.
- hcp_job_specialty: Lookup for HCP specialties.
- hcp_level_of_expertise: Lookup for HCP expertise levels.
- respondent_hcp_job_title: Links respondents to HCP job titles.
- respondent_hcp_specialty: Links respondents to HCP specialties, includes hcp_number, 
  hcp_sub_specialty.
- respondent_hcp_level_of_expertise: Links respondents to expertise levels.

- socio_economic: Socio-economic profile per respondent linked by email. Fields include 
  job_status, current_job_title, job_title_tier, industry, highest_education_level, 
  annual_household_income.

- household: Household info per respondent linked by email. Fields include relationship, 
  children, number_of_children.

- clients: Client companies (client_id, client_name, created_date).

- projects: Research projects (project_number PK, project_name, client_id, topic, 
  project_type, project_state, created_date, end_date).

- project_respondent: Links respondents to projects via email and project_number. 
  Includes incentive, currency, interaction_level, last_activity_date.

- survey_response: Survey answers per respondent per project. Fields include email, 
  project_number, survey_question, survey_answer.

- mailings: Email campaigns (mailing_id, mailing_name).

- messages: Individual emails sent (system_message_id PK, mailing_id, subject, from_email).

- delivery_logs: Delivery status per message per recipient. Fields include system_message_id, 
  recipient_email, status, status_date, failure_type, failure_code, reason.

- engagement: Email engagement per message. Fields include system_message_id, first_open, 
  last_open, open_count, first_click, last_click, click_count, last_event_type.

- mx_records: MX provider info per domain.

- company_profile: B2B respondent company info linked by email. Fields include company_name, 
  company_size, company_turnover, years_in_business, industry, approximate_salary_bracket.

- providers / servers: Email sending infrastructure lookup tables.

CRITICAL RULES YOU MUST ALWAYS FOLLOW:
1. ALWAYS exclude emails that appear in the unsubscribe_blacklist table from ANY query 
   result that returns respondent emails or counts. Always use:
   AND email NOT IN (SELECT email FROM unsubscribe_blacklist)
   or a LEFT JOIN with WHERE unsubscribe_blacklist.email IS NULL.
2. Never query staging tables (staging_emails, staging_respondents, staging_projects, 
   staging_respondent_projects).
3. Never query the error_log table.
4. Always use lowercase table and column names.
5. Use PostgreSQL syntax only.
"""

# ----------------------------
# SQL generation prompt
# ----------------------------
SQL_PROMPT = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
{schema}

Given the database schema and rules above, write a PostgreSQL SQL query to answer this question:
{question}

Return ONLY the SQL query with no explanation, no markdown, no code fences.
If the question cannot be answered with SQL, return the word: UNSUPPORTED
"""
)

# ----------------------------
# Response format prompt
# ----------------------------
RESPONSE_PROMPT = PromptTemplate(
    input_variables=["question", "data"],
    template="""
You are a helpful data analyst. A user asked: "{question}"

The query returned this data:
{data}

If the result is a single value or a simple yes/no fact, respond in one clear sentence.
If the result has multiple rows or columns, present it as a clean markdown table.
Do not add unnecessary commentary. Be concise and professional.
"""
)

# ----------------------------
# Core functions
# ----------------------------
def run_query(sql: str) -> pd.DataFrame | None:
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df
    except Exception as e:
        st.error(f"SQL execution error: {e}")
        return None

def generate_sql(question: str) -> str | None:
    llm = get_llm()
    chain = SQL_PROMPT | llm
    result = chain.invoke({"schema": SCHEMA_DESCRIPTION, "question": question}).content
    result = result.strip()
    if result == "UNSUPPORTED" or not result.lower().startswith("select"):
        return None
    return result

def generate_response(question: str, df: pd.DataFrame) -> str | None:
    llm = get_llm()
    chain = RESPONSE_PROMPT | llm
    data_str = df.to_string(index=False) if df is not None else "No data returned."
    return chain.invoke({"question": question, "data": data_str}).content

# ----------------------------
# Cached wrappers (matching app.py signatures exactly)
# ----------------------------
@st.cache_data(show_spinner="Generating sample questions ...")
def generate_questions_cached():
    return [
        "How many active respondents do we have?",
        "Which clients have the most projects?",
        "How many respondents are healthcare professionals?",
        "What are the top 5 conditions in our panel?",
        "Show me all projects currently in progress",
        "How many respondents have unsubscribed?",
    ]

@st.cache_data(show_spinner="Generating SQL query ...")
def generate_sql_cached(question: str):
    return generate_sql(question)

@st.cache_data(show_spinner="Checking for valid SQL ...")
def is_sql_valid_cached(sql: str):
    return sql is not None and sql.strip().lower().startswith("select")

@st.cache_data(show_spinner="Running SQL query ...")
def run_sql_cached(sql: str):
    return run_query(sql)

@st.cache_data(show_spinner="Checking if we should generate a chart ...")
def should_generate_chart_cached(question, sql, df):
    return False  # charts dropped for prototype

@st.cache_data(show_spinner="Generating Plotly code ...")
def generate_plotly_code_cached(question, sql, df):
    return None

@st.cache_data(show_spinner="Running Plotly code ...")
def generate_plot_cached(code, df):
    return None

@st.cache_data(show_spinner="Generating followup questions ...")
def generate_followup_cached(question, sql, df):
    return []

@st.cache_data(show_spinner="Generating summary ...")
def generate_summary_cached(question, df):
    if df is None or df.empty:
        return "No data was returned for that question."
    return generate_response(question, df)
