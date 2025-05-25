import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector
import os
from dotenv import load_dotenv
import re

# Load environment variables
load_dotenv()

# Snowflake/Cortex Configuration
HOST = "bnkzyio-ljb86662.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_TIMEOUT = 360000  # 6 minutes in milliseconds
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.Grants_search_services"  # Placeholder
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/GRANTSyaml.yaml'  # Placeholder
SNOWFLAKE_USER="cortex"
SNOWFLAKE_PASSWORD="Dilytics@12345"
SNOWFLAKE_ROLE="ACCOUNTADMIN"
SNOWFLAKE_WAREHOUSE="COMPUTE_WH"

# Streamlit page configuration
st.set_page_config(
    page_title="Cortex AI Assistant for Grants",
    layout="wide",
    initial_sidebar_state="auto"
)

# Hide Streamlit branding
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
[data-testid="stChatMessage"] {
    opacity: 1 !important;
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "current_sql" not in st.session_state:
    st.session_state.current_sql = None
if "chart_x_axis" not in st.session_state:
    st.session_state.chart_x_axis = None
if "chart_y_axis" not in st.session_state:
    st.session_state.chart_y_axis = None
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar Chart"

# Snowflake connection
@st.cache_resource
def init_snowflake_connection():
    required_env_vars = [
        "SNOWFLAKE_USER",
        "SNOWFLAKE_PASSWORD",
        "SNOWFLAKE_ROLE",
        "SNOWFLAKE_WAREHOUSE"
    ]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
    
    try:
        conn = snowflake.connector.connect(
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            account=HOST.replace(".snowflakecomputing.com", ""),
            role=os.getenv("SNOWFLAKE_ROLE"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database=DATABASE,
            schema=SCHEMA
        )
        return conn
    except Exception as e:
        raise Exception(f"Failed to connect to Snowflake: {str(e)}")

# Initialize connection
try:
    conn = init_snowflake_connection()
    session = conn.cursor()
except Exception as e:
    st.error(str(e))
    if st.session_state.debug_mode:
        st.write(f"Debug: Environment variables - USER: {os.getenv('SNOWFLAKE_USER')}, ROLE: {os.getenv('SNOWFLAKE_ROLE')}, WAREHOUSE: {os.getenv('SNOWFLAKE_WAREHOUSE')}, HOST: {HOST}")
    st.stop()

# Utility functions
def run_snowflake_query(query):
    try:
        if not query:
            return None, "⚠️ No SQL query generated."
        session.execute(query)
        df = pd.DataFrame(session.fetchall(), columns=[col[0] for col in session.description])
        return df, None
    except Exception as e:
        return None, f"❌ SQL Execution Error: {str(e)}"

def is_structured_query(query: str):
    structured_patterns = [
        r'\b(total|count|sum|avg|max|min|group by|order by|where|award|budget|encumbrance)\b'
    ]
    return any(re.search(pattern, query.lower()) for pattern in structured_patterns)

def display_chart_tab(df: pd.DataFrame, prefix: str = "chart", query: str = ""):
    if df.empty or len(df.columns) < 2:
        return
    query_lower = query.lower()
    default_chart = "Bar Chart"
    if "date" in query_lower:
        default_chart = "Line Chart"
    elif "award" in query_lower:
        default_chart = "Pie Chart"
    
    all_cols = list(df.columns)
    col1, col2, col3 = st.columns(3)
    
    default_x = st.session_state.get(f"{prefix}_x", all_cols[0])
    x_index = all_cols.index(default_x) if default_x in all_cols else 0
    x_col = col1.selectbox("X-axis", all_cols, index=x_index, key=f"{prefix}_x")
    
    remaining_cols = [c for c in all_cols if c != x_col]
    default_y = remaining_cols[0] if remaining_cols else all_cols[0]
    y_index = remaining_cols.index(default_y) if remaining_cols and default_y in remaining_cols else 0
    y_col = col2.selectbox("Y-axis", remaining_cols, index=y_index, key=f"{prefix}_y")
    
    chart_options = ["Line Chart", "Bar Chart", "Pie Chart"]
    default_type = chart_options.index(default_chart) if default_chart in chart_options else 0
    chart_type = col3.selectbox("Chart Type", chart_options, index=default_type, key=f"{prefix}_type")
    
    if chart_type == "Line Chart":
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        st.plotly_chart(fig, key=f"{prefix}_line")
    elif chart_type == "Bar Chart":
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        st.plotly_chart(fig, key=f"{prefix}_bar")
    elif chart_type == "Pie Chart":
        fig = px.pie(df, names=x_col, values=y_col, title=f"{y_col} by {x_col}")
        st.plotly_chart(fig, key=f"{prefix}_pie")

# Sidebar
with st.sidebar:
    st.image("https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg", width=100)
    st.session_state.debug_mode = st.checkbox("Enable Debug Mode")
    if st.button("New Conversation"):
        st.session_state.chat_history = []
        st.session_state.current_query = None
        st.session_state.current_results = None
        st.session_state.current_sql = None
        st.rerun()
    
    st.subheader("Sample Questions")
    sample_questions = [
        "What is the total actual award budget posted?",
        "What is the total actual award posted?",
        "What is the total amount of award encumbrances approved?",
        "What is the total task actual posted by award name?"
    ]
    for q in sample_questions:
        if st.button(q, key=f"sample_{hash(q)}"):
            st.session_state.current_query = q

# Main UI
st.title("Cortex AI Assistant for Grants")
st.markdown(f"Semantic Model: `{SEMANTIC_MODEL.split('/')[-1]}`")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "results" in message and message["results"] is not None:
            with st.expander("View SQL Query"):
                st.code(message["sql"], language="sql")
            st.markdown(f"**Query Results ({len(message['results'])} rows):**")
            st.dataframe(message["results"])
            if not message["results"].empty and len(message["results"].columns) >= 2:
                st.markdown("**📈 Visualization:**")
                display_chart_tab(message["results"], prefix=f"chart_{hash(message['content'])}", query=message["content"])

# Query input
query = st.chat_input("Ask your question...")
if query or st.session_state.current_query:
    query = query or st.session_state.current_query
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            is_structured = is_structured_query(query)
            assistant_response = {"role": "assistant", "content": ""}
            
            if is_structured:
                query_map = {
                    "What is the total actual award budget posted?": 
                        "SELECT SUM(BUDGET) AS TOTAL_BUDGET FROM AI.DWH_MART.GRANTS",
                    "What is the total actual award posted?": 
                        "SELECT SUM(ACTUAL_POSTED) AS TOTAL_ACTUAL FROM AI.DWH_MART.GRANTS",
                    "What is the total amount of award encumbrances approved?": 
                        "SELECT SUM(ENCUMBRANCE_APPROVED) AS TOTAL_APPROVED FROM AI.DWH_MART.GRANTS",
                    "What is the total task actual posted by award name?": 
                        "SELECT AWARD_NAME, SUM(ACTUAL_POSTED) AS TOTAL_ACTUAL FROM AI.DWH_MART.GRANTS GROUP BY AWARD_NAME"
                }
                sql = query_map.get(query, f"SELECT * FROM AI.DWH_MART.GRANTS WHERE QUERY_TEXT ILIKE '%{query}%' LIMIT 10")
                if st.session_state.debug_mode:
                    st.write(f"Debug: SQL: {sql}")
                st.markdown("**SQL Query:**")
                st.code(sql, language="sql")
                results_df, query_error = run_snowflake_query(sql)
                if query_error:
                    st.error(query_error)
                    assistant_response["content"] = query_error
                elif results_df is not None and not results_df.empty:
                    response_content = f"**Results ({len(results_df)} rows):**"
                    st.markdown(response_content)
                    st.dataframe(results_df)
                    if len(results_df.columns) >= 2:
                        st.markdown("**📈 Visualization:**")
                        display_chart_tab(results_df, prefix=f"chart_{hash(query)}", query=query)
                    assistant_response.update({
                        "content": response_content,
                        "sql": sql,
                        "results": results_df
                    })
                else:
                    response_content = "⚠️ No data found."
                    st.markdown(response_content)
                    assistant_response["content"] = response_content
            else:
                response_content = f"Summary for '{query}': Placeholder response due to Cortex Search unavailability."
                st.markdown(response_content)
                assistant_response["content"] = response_content
            
            st.session_state.chat_history.append(assistant_response)
            st.session_state.current_query = None
