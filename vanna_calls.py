import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
import time

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
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=st.secrets["GROQ_API_KEY"],
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
6. You can disregard is_deleted and is_active in everyday queries unless specified in the question.
7. When you run a SQL query that returns data, DO NOT generate a Markdown table of the results in your text response. 
   The user interface will automatically display the data. 
   Your text response should only be a brief summary of what you found, never the raw rows themselves.
8. Several columns in the database are PostgreSQL enum types, not plain text. 
   These include but are not limited to: country, uk_region, county_state, gender, 
   ethnicity, relationship, job_status, job_title_tier, industry, 
   highest_education_level, annual_household_income, company_size, company_turnover, 
   years_in_business, approximate_salary_bracket, project_state, company_turnover.
   
   For ANY column that filters by a categorical or descriptive value, NEVER assume 
   the format or use abbreviations. Always use the full stored value exactly as it 
   appears in the database. For example: 'United States of America' not 'US', 
   'United Kingdom' not 'UK', 'Male' not 'M'.
   
   When unsure of the exact enum value, use ILIKE for partial matching instead:
   WHERE column::text ILIKE '%keyword%'
   This casts the enum to text first which avoids type errors entirely.
9. ALWAYS qualify every column name with its table alias when writing JOIN queries.
   Never write SELECT email, SELECT country etc when multiple tables are joined.
   Always write SELECT r.email, SELECT a.country etc.
   This applies to WHERE clauses, ON clauses, GROUP BY, and ORDER BY as well.
   Example: WHERE r.email NOT IN (...) not WHERE email NOT IN (...).
10.When a question asks for a LIST of people or records, always include at minimum:
   r.email, r.first_name, r.last_name in the SELECT. Never return email alone 
   as a list — it is not human readable enough.
11.When a question asks to COUNT something, return a single aliased column.
    Example: SELECT COUNT(DISTINCT r.email) AS total_respondents
    Never return an unnamed count column.
12.When joining respondent to addresses, always use LEFT JOIN not INNER JOIN unless 
    the question specifically requires an address field to be present. Many respondents 
    may not have an address record and an INNER JOIN would silently exclude them from 
    counts.
13.Never use SELECT * in any query. Always specify the columns you need explicitly.
14. When filtering by date, always use TIMESTAMP WITH TIME ZONE safe comparisons.
    Example: WHERE created_date >= '2024-01-01'::timestamptz
    Never assume a date column is plain DATE type.
"""

# ----------------------------
# SQL generation prompt
# ----------------------------
SQL_PROMPT = PromptTemplate(
    input_variables=["schema", "history", "question"],
    template="""
{schema}

{history}
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
You are a data analyst reporting internal database results to a colleague.
A user asked: "{question}"

The query returned this data:
{data}

Rules:
- Report ONLY what the data shows. Never compare to external benchmarks or real world statistics.
- If the result is a single value, respond in one short sentence stating just the number.
- If the result has multiple rows or columns, present it as a clean markdown table.
- Do not add commentary, caveats, or explanations unless the data is empty.
- When you run a SQL query that returns data, DO NOT generate a Markdown table of the results in your text response. 
  The user interface will automatically display the data. Your text response should only be a brief summary of what you found, never the raw rows themselves.
- If no data was returned, say: "No results were found for that question."
"""
)

# ----------------------------
# Core functions
# ----------------------------
def run_query(sql: str, max_retries: int = 5, delay: int = 3) -> pd.DataFrame | None:
    """Runs the query with a retry loop to handle Neon's cold starts."""
    engine = get_engine()
    df = None  # Initialize df here
    
    # st.status provides a spinner UI that we can update text for dynamically
    with st.status("Connecting to database...", expanded=True) as status:
        for attempt in range(max_retries):
            try:
                with engine.connect() as conn:
                    status.update(label="Executing query...", state="running")
                    result = conn.execute(text(sql))
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    status.update(label="Query successful!", state="complete", expanded=False)
                    break  # Exit the retry loop gracefully instead of returning early
                    
            except Exception as e:
                error_str = str(e).lower()
                # Check if the error is likely due to the database being asleep
                is_conn_error = any(keyword in error_str for keyword in [
                    "connection", "timeout", "closed", "ssl", "operationalerror"
                ])
                
                if is_conn_error and attempt < max_retries - 1:
                    status.update(
                        label=f"Waking up database... (Attempt {attempt + 1} of {max_retries})", 
                        state="running"
                    )
                    time.sleep(delay)  # Wait a few seconds before trying again
                else:
                    # If it's a strict SQL syntax error or we ran out of retries, fail out
                    status.update(label="Query failed.", state="error")
                    st.session_state["last_sql_error"] = str(e)
                    st.error(f"SQL execution error: {e}")
                    break  # Also break here on complete failure
                    
    # Return df OUTSIDE the context manager so Streamlit fully closes the UI status
    return df

def get_column_samples(sql: str) -> str:
    """Looks at the SQL, finds the tables used, and fetches distinct values for text columns."""
    engine = get_engine()
    samples = []
    
    try:
        # Extract table names crudely from the SQL
        words = sql.lower().split()
        tables = []
        for i, word in enumerate(words):
            if word in ("from", "join") and i + 1 < len(words):
                table = words[i + 1].strip("(),;")
                if table and not table.startswith("("):
                    tables.append(table)
        
        with engine.connect() as conn:
            for table in set(tables):
                try:
                    # Get text columns for this table
                    col_result = conn.execute(text(f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = '{table}' 
                        AND data_type IN ('text', 'character varying', 'USER-DEFINED')
                        LIMIT 10
                    """))
                    columns = [row[0] for row in col_result]
                    
                    for col in columns:
                        try:
                            val_result = conn.execute(text(f"""
                                SELECT DISTINCT {col} 
                                FROM {table} 
                                WHERE {col} IS NOT NULL 
                                LIMIT 10
                            """))
                            values = [str(row[0]) for row in val_result]
                            if values:
                                samples.append(f"{table}.{col}: {', '.join(values)}")
                        except:
                            pass
                except:
                    pass
    except:
        pass
    
    return "\n".join(samples)

def generate_sql_with_retry(question: str, history: str = "") -> str | None:
    """Generates SQL, runs it, and if empty retries with real column values."""
    
    with st.expander("🔍 Query Process", expanded=True):
        # Step 1
        st.markdown("**Step 1: Generating SQL...**")
        sql = generate_sql(question, history=history)
        
        if not sql:
            st.error("Could not generate a valid SQL query.")
            return None
        
        st.code(sql, language="sql")
        
        # Step 2
        st.markdown("**Step 2: Running query...**")
        df = run_query(sql)
        sql_error = st.session_state.pop("last_sql_error", None)
        
        if df is not None and not df.empty:
            st.success(f"Query returned {len(df)} row(s). No retry needed.")
            return sql
        
        # Step 3 — retry
        if (df is not None and df.empty) or (df is None and sql_error):
            if sql_error:
                st.warning(f"Query failed with error: {sql_error}")
            else:
                st.warning("Query returned 0 results. Fetching actual column values to retry...")
            
            st.markdown("**Step 3: Fetching real column values from database...**")
            samples = get_column_samples(sql)
            
            if samples:
                st.text(samples)
                st.markdown("**Step 4: Retrying with correct values...**")
                
                llm = get_llm()
                retry_prompt = PromptTemplate(
                    input_variables=["schema", "history", "question", "bad_sql", "samples"],
                    template="""
{schema}

{history}
You previously generated this SQL query:
{bad_sql}

It returned zero results. Here are the actual distinct values stored in the relevant columns:
{samples}

Using these exact values, rewrite the SQL query to answer this question:
{question}

Return ONLY the SQL query with no explanation, no markdown, no code fences.
"""
                )
                chain = retry_prompt | llm
                sql = chain.invoke({
                    "schema": SCHEMA_DESCRIPTION,
                    "history": history,
                    "question": question,
                    "bad_sql": sql,
                    "samples": samples
                }).content.strip()
                
                st.markdown("**Retried SQL:**")
                st.code(sql, language="sql")
            else:
                st.warning("Could not fetch column samples for retry.")
    
    return sql

def generate_sql(question: str) -> str | None:
    llm = get_llm()
    chain = SQL_PROMPT | llm
    result = chain.invoke({"schema": SCHEMA_DESCRIPTION, "history": history, "question": question}).content
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

def generate_sql_cached(question: str, history: str = ""):
    return generate_sql_with_retry(question, history=history)

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

    if len(df) == 1:
        return generate_response(question, df)
    elif len(df) <= 5:
        return generate_response(question, df.head(5))
    else:
        # For large result sets, just confirm what was returned
        cols = ", ".join(df.columns.tolist())
        return f"Query returned {len(df)} records with columns: {cols}."
    return generate_response(question, df)
