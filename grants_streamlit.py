import streamlit as st
import json
import re
import requests
import snowflake.connector
import pandas as pd
from snowflake.snowpark import Session
from snowflake.core import Root
from typing import Any, Dict, List, Optional, Tuple
import plotly.express as px
import time
import uuid

# --- Snowflake/Cortex Configuration ---
HOST = "itvdwya-azb15149.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.GRANTS_SEARCH_SERVICES"
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/grantsyaml_27.yaml'

# --- Model Options ---
MODELS = [
    "mistral-large",
    "snowflake-arctic",
    "llama3-70b",
    "llama3-8b",
]

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Cortex AI-Grants Assistant",
    layout="wide",
    initial_sidebar_state="auto"
)

# --- Session State Initialization ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.CONN = None
    st.session_state.snowpark_session = None
    st.session_state.chat_history = []
    st.session_state.messages = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "last_suggestions" not in st.session_state:
    st.session_state.last_suggestions = []
if "chart_x_axis" not in st.session_state:
    st.session_state.chart_x_axis = None
if "chart_y_axis" not in st.session_state:
    st.session_state.chart_y_axis = None
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar Chart"
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "current_results" not in st.session_state:
    st.session_state.current_results = None
if "current_sql" not in st.session_state:
    st.session_state.current_sql = None
if "current_summary" not in st.session_state:
    st.session_state.current_summary = None
if "service_metadata" not in st.session_state:
    st.session_state.service_metadata = [{"name": "AI.DWH_MART.GRANTS_SEARCH_SERVICES", "search_column": ""}]
if "selected_cortex_search_service" not in st.session_state:
    st.session_state.selected_cortex_search_service = "AI.DWH_MART.GRANTS_SEARCH_SERVICES"
if "model_name" not in st.session_state:
    st.session_state.model_name = "mistral-large"
if "num_retrieved_chunks" not in st.session_state:
    st.session_state.num_retrieved_chunks = 100
if "num_chat_messages" not in st.session_state:
    st.session_state.num_chat_messages = 10
if "use_chat_history" not in st.session_state:
    st.session_state.use_chat_history = True
if "clear_conversation" not in st.session_state:
    st.session_state.clear_conversation = False
if "show_selector" not in st.session_state:
    st.session_state.show_selector = False
if "show_greeting" not in st.session_state:
    st.session_state.show_greeting = True
if "data_source" not in st.session_state:
    st.session_state.data_source = "Database"
if "show_about" not in st.session_state:
    st.session_state.show_about = False
if "show_help" not in st.session_state:
    st.session_state.show_help = False
if "show_history" not in st.session_state:
    st.session_state.show_history = False
if "query" not in st.session_state:
    st.session_state.query = None
if "previous_query" not in st.session_state:
    st.session_state.previous_query = None
if "previous_sql" not in st.session_state:
    st.session_state.previous_sql = None
if "previous_results" not in st.session_state:
    st.session_state.previous_results = None
if "show_sample_questions" not in st.session_state:
    st.session_state.show_sample_questions = False

# --- CSS Styling ---
st.markdown("""
<style>
* {
    box-sizing: border-box;
    font-family: 'Roboto', 'Arial', sans-serif !important;
}

/* Hide Streamlit's default elements */
#MainMenu, header, footer {
    visibility: hidden;
}

/* Global app container */
.stApp {
    padding-top: 80px !important;
}

/* Fixed header styling */
.fixed-header {
    position: fixed;
    top: 0;
    left: 20px;
    right: 20px;
    z-index: 999;
    background-color: #ffffff;
    padding: 8px 10px;
    text-align: center;
    pointer-events: none;
}
.fixed-header h1 {
    font-size: 28px !important;
    color: #29B5E8;
    margin: 0 0 2px 0 !important;
    font-weight: 600 !important;
}
.fixed-header p {
    font-size: 16px !important;
    color: #333;
    margin: 0 !important;
    font-weight: 400 !important;
}
.fixed-header a {
    pointer-events: none !important;
    text-decoration: none !important;
    color: inherit !important;
    cursor: default !important;
}

/* Dilytics logo */
.dilytics-logo {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
    width: 150px;
    height: auto;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    opacity: 1 !important;
    background-color: transparent !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow: hidden !important;
    margin-bottom: 10px !important;
    padding: 8px !important;
    border-radius: 8px !important;
    text-align: left !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
    width: 100% !important;
}
[data-testid="stChatMessageContent"] {
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    overflow: hidden !important;
    width: 100% !important;
    max-width: 100% !important;
    text-align: left !important;
    padding-left: 0 !important;
    margin-left: 0 !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
}

/* Dataframes */
[data-testid="stDataFrame"] {
    text-align: left !important;
    width: 100% !important;
    margin: 0 auto !important;
    font-size: 14px !important;
    line-height: 1.4 !important;
}

/* Markdown elements */
[data-testid="stMarkdown"] {
    text-align: left !important;
    width: 100% !important;
    padding: 3px 0 !important;
    font-size: 15px !important;
    line-height: 1.5 !important;
}
[data-testid="stMarkdown"] h1, [data-testid="stMarkdown"] h2, [data-testid="stMarkdown"] h3 {
    font-weight: 600 !important;
    margin: 5px 0 !important;
}

/* Hide copy buttons */
.copy-button, [data-testid="copy-button"], [title="Copy to clipboard"], [data-testid="stTextArea"] {
    display: none !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    padding: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stButton"] > button {
    background-color: #29B5E8 !important;
    color: white !important;
    font-weight: 600 !important;
    width: 100% !important;
    border-radius: 0px !important;
    margin: 2px 0 !important;
    border: none !important;
    padding: 0.5rem 1rem !important;
    font-size: 14px !important;
    line-height: 1.4 !important;
}
[data-testid="stSidebar"] [data-testid="stButton"][aria-label="Clear conversation"] > button,
[data-testid="stSidebar"] [data-testid="stButton"][aria-label="About"] > button,
[data-testid="stSidebar"] [data-testid="stButton"][aria-label="Help & Documentation"] > button,
[data-testid="stSidebar"] [data-testid="stButton"][aria-label="History"] > button,
[data-testid="stSidebar"] [data-testid="stButton"][aria-label="Sample Questions"] > button {
    background-color: #28A745 !important;
    color: white !important;
    font-weight: 500 !important;
    border: 1px solid #28A745 !important;
    font-size: 14px !important;
    line-height: 1.4 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] {
    margin: 5px 0 !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdown"] {
    font-size: 14px !important;
    line-height: 1.4 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Add Logo ---
if st.session_state.authenticated:
    st.markdown(
        f'<img src="https://raw.githubusercontent.com/nkumbala129/30-05-2025/main/Dilytics_logo.png" class="dilytics-logo">',
        unsafe_allow_html=True
    )

# --- Utility Functions ---
def stream_text(text: str, chunk_size: int = 2, delay: float = 0.04):
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        time.sleep(delay)

def submit_grant_application(grant_id: str, applicant_name: str, application_details: str):
    try:
        application_id = str(uuid.uuid4())
        insert_query = """
        INSERT INTO GRANT_APPLICATIONS 
        (APPLICATION_ID, GRANT_ID, APPLICANT_NAME, APPLICATION_DETAILS, SUBMITTED_AT, STATUS)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP(), 'PENDING')
        """
        session.sql(insert_query).bind((application_id, grant_id, applicant_name, application_details)).collect()
        return True, f"📝 Grant application submitted successfully! Application ID: {application_id}"
    except Exception as e:
        return False, f"❌ Failed to submit grant application: {str(e)}"

def start_new_conversation():
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.current_query = None
    st.session_state.current_results = None
    st.session_state.current_sql = None
    st.session_state.current_summary = None
    st.session_state.chart_x_axis = None
    st.session_state.chart_y_axis = None
    st.session_state.chart_type = "Bar Chart"
    st.session_state.last_suggestions = []
    st.session_state.clear_conversation = False
    st.session_state.show_greeting = True
    st.session_state.query = None
    st.session_state.show_history = False
    st.session_state.previous_query = None
    st.session_state.previous_sql = None
    st.session_state.previous_results = None
    st.session_state.message_counter = 0  # Reset counter
    st.rerun()

def init_service_metadata():
    if not st.session_state.service_metadata:
        st.session_state.service_metadata = [{"name": "AI.DWH_MART.GRANTS_SEARCH_SERVICES", "search_column": ""}]
    try:
        svc_search_col = session.sql("DESC CORTEX SEARCH SERVICE AI.DWH_MART.GRANTS_SEARCH_SERVICES;").collect()[0]["search_column"]
        st.session_state.service_metadata = [{"name": "AI.DWH_MART.GRANTS_SEARCH_SERVICES", "search_column": svc_search_col}]
    except Exception as e:
        st.error(f"❌ Failed to verify AI.DWH_MART.GRANTS_SEARCH_SERVICES: {str(e)}")

def query_cortex_search_service(query):
    try:
        db, schema = session.get_current_database(), session.get_current_schema()
        root = Root(session)
        cortex_search_service = (
            root.databases[db]
            .schemas[schema]
            .cortex_search_services["AI.DWH_MART.GRANTS_SEARCH_SERVICES"]
        )
        context_documents = cortex_search_service.search(
            query, columns=[], limit=st.session_state.num_retrieved_chunks
        )
        results = context_documents.results
        service_metadata = st.session_state.service_metadata
        search_col = service_metadata[0]["search_column"]
        context_str = ""
        for i, r in enumerate(results):
            context_str += f"Context document {i+1}: {r[search_col]} \n"
        return context_str
    except Exception as e:
        st.error(f"❌ Error querying Cortex Search service: {str(e)}")
        return ""

def get_chat_history():
    start_index = max(0, len(st.session_state.chat_history) - st.session_state.num_chat_messages)
    return st.session_state.chat_history[start_index : len(st.session_state.chat_history) - 1]

def make_chat_history_summary(chat_history, question):
    chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    prompt = f"""
        [INST]
        You are a conversational AI assistant. Based on the chat history below and the current question, generate a single, clear, and concise query that combines the context of the chat history with the current question. The resulting query should be in natural language and should reflect the user's intent in the conversational flow. Ensure the query is standalone and can be understood without needing to refer back to the chat history.

        For example:
        - If the chat history contains "user: Total number of grants awarded?" and the current question is "by year", the resulting query should be "What is the total number of grants awarded by year?"

        Answer with only the query. Do not add any explanation.

        <chat_history>
        {chat_history_str}
        </chat_history>
        <question>
        {question}
        </question>
        [/INST]
    """
    summary = complete(st.session_state.model_name, prompt)
    if st.session_state.debug_mode:
        st.sidebar.text_area("Chat History Summary", summary.replace("$", "\$"), height=150)
    return summary

def create_prompt(user_question):
    chat_history_str = ""
    if st.session_state.use_chat_history:
        chat_history = get_chat_history()
        if chat_history:
            question_summary = make_chat_history_summary(chat_history, user_question)
            prompt_context = query_cortex_search_service(question_summary)
            chat_history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
        else:
            prompt_context = query_cortex_search_service(user_question)
    else:
        prompt_context = query_cortex_search_service(user_question)
        chat_history = []
    
    if not prompt_context.strip():
        return complete(st.session_state.model_name, user_question)
    
    prompt = f"""
        [INST]
        You are a helpful AI chat assistant for grants management. Use the provided context and chat history to provide a coherent, concise, and relevant answer to the user's question.
        <chat_history>
        {chat_history_str}
        </chat_history>
        <context>
        {prompt_context}
        </context>
        <question>
        {user_question}
        </question>
        [/INST]
        Answer:
    """
    return complete(st.session_state.model_name, prompt)

def get_user_questions(limit=10):
    user_questions = [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
    return user_questions[-limit:][::-1]

# --- Main Application Logic ---
if not st.session_state.authenticated:
    st.title("Welcome to Snowflake Cortex AI")
    st.markdown("Please login to interact with your grants data")
    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")
    if st.button("Login"):
        try:
            conn = snowflake.connector.connect(
                user="CORTEX",
                password="Dilytics@12345",
                account="itvdwya-azb15149",
                host=HOST,
                port=443,
                warehouse="COMPUTE_WH",
                role="ACCOUNTADMIN",
                database=DATABASE,
                schema=SCHEMA,
            )
            st.session_state.CONN = conn
            snowpark_session = Session.builder.configs({"connection": conn}).create()
            st.session_state.snowpark_session = snowpark_session
            with conn.cursor() as cur:
                cur.execute("USE DATABASE AI")
                cur.execute("USE SCHEMA DWH_MART")
                cur.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = TRUE")
            st.session_state.authenticated = True
            st.success("Authentication successful!")
            st.rerun()
        except Exception as e:
            st.error(f"Authentication failed: {e}")
else:
    # Initialize message_counter after authentication
    if "message_counter" not in st.session_state:
        st.session_state.message_counter = 0
        if st.session_state.debug_mode:
            st.sidebar.write("Debug: Initialized message_counter to 0")

    session = st.session_state.snowpark_session
    root = Root(session)

    def run_snowflake_query(query):
        try:
            if not query:
                return None
            df = session.sql(query)
            data = df.collect()
            if not data:
                return None
            columns = df.columns
            result_df = pd.DataFrame(data, columns=columns)
            # Debug: Print column names to identify the correct award number and date columns
            if st.session_state.debug_mode:
                st.write("Debug - DataFrame Columns:", result_df.columns.tolist())
            # Convert award number columns to strings to prevent 'k' formatting
            award_number_candidates = ["AWARD_NUMBER", "AWARD_ID", "GRANT_ID", "AWARD_NO"]
            for col in award_number_candidates:
                if col in result_df.columns:
                    result_df[col] = result_df[col].astype(str)
            # Format date columns to YYYY-MM-DD
            date_column_candidates = ["DATE", "AWARD_DATE", "SUBMITTED_AT", "DW_DATE_ALLOCATED_KEY"]
            for col in result_df.columns:
                if col in date_column_candidates or "DATE" in col.upper():
                    try:
                        result_df[col] = pd.to_datetime(result_df[col]).dt.strftime('%Y-%m-%d')
                    except (ValueError, TypeError):
                        continue  # Skip if the column can't be converted to datetime
            return result_df
        except Exception as e:
            st.error(f"Error executing SQL query: {str(e)}")
            return None

    def is_structured_query(query: str):
        structured_patterns = [
            r'\b(count|number|amount|where|group by|order by|sum|avg|max|min|total|how many|which|show|list)\b',
            r'\b(grant|award|funding|recipient|application|status|budget|encumbrance|posted|approved|task|actual)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in structured_patterns)

    def is_complete_query(query: str):
        complete_patterns = [r'\b(generate|write|create|describe|explain)\b']
        return any(re.search(pattern, query.lower()) for pattern in complete_patterns)

    def is_summarize_query(query: str):
        summarize_patterns = [r'\b(summarize|summary|condense)\b']
        return any(re.search(pattern, query.lower()) for pattern in summarize_patterns)

    def is_question_suggestion_query(query: str):
        suggestion_patterns = [
            r'\b(what|which|how)\b.*\b(questions|queries)\b.*\s+\b(ask|can i ask)\b',
            r'\b(give me|show me|list)\b.*\s+\b(questions|examples)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in suggestion_patterns)

    def is_greeting_query(query: str):
        greeting_patterns = [r'^\s*(hello|hi|hey|greet)\s*$']
        return any(re.search(pattern, query.lower()) for pattern in greeting_patterns)

    def complete(model, prompt):
        try:
            prompt = prompt.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt}') AS response"
            result = session.sql(query).collect()
            return result[0]["RESPONSE"]
        except Exception as e:
            st.error(f"Error in COMPLETE function: {str(e)}")
            return None

    def summarize(text):
        try:
            text = text.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
            result = session.sql(query).collect()
            return result[0]["SUMMARY"]
        except Exception as e:
            st.error(f"Error in SUMMARIZE function: {str(e)}")
            return None

    def parse_sse_response(response_text: str) -> List[Dict]:
        events = []
        lines = response_text.strip().split("\n")
        current_event = {}
        for line in lines:
            if line.startswith("event:"):
                current_event["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                if data_str != "[DONE]":
                    try:
                        data_json = json.loads(data_str)
                        current_event["data"] = data_json
                        events.append(current_event)
                        current_event = {}
                    except json.JSONDecodeError as e:
                        st.error(f"Failed to parse SSE data: {str(e)}")
        return events

    def process_sse_response(response, is_structured):
        sql = ""
        search_results = []
        if not response:
            return sql, search_results
        for event in response:
            if event.get("event") == "message.delta" and "data" in event:
                delta = event["data"].get("delta", {})
                content = delta.get("content", [])
                for item in content:
                    if item.get("type") == "tool_results":
                        tool_results = item.get("tool_results", {})
                        if "content" in tool_results:
                            for result in tool_results["content"]:
                                if result.get("type") == "json":
                                    result_data = result.get("json", {})
                                    if is_structured and "sql" in result_data:
                                        sql = result_data.get("sql", "")
                                    elif not is_structured and "searchResults" in result_data:
                                        search_results = [sr["text"] for sr in result_data["searchResults"]]
        return sql.strip(), search_results

    def snowflake_api_call(query: str, is_structured: bool = False):
        payload = {
            "model": st.session_state.model_name,
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "tools": []
        }
        if is_structured:
            payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
            payload["tool_resources"] = {"analyst1": {"semantic_model_file": SEMANTIC_MODEL}}
        else:
            payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
            payload["tool_resources"] = {"search1": {"name": st.session_state.selected_cortex_search_service, "max_results": st.session_state.num_retrieved_chunks}}
        try:
            resp = requests.post(
                url=f"https://{HOST}{API_ENDPOINT}",
                json=payload,
                headers={
                    "Authorization": f'Snowflake Token="{st.session_state.CONN.rest.token}"',
                    "Content-Type": "application/json",
                },
                timeout=API_TIMEOUT // 1000
            )
            if resp.status_code < 400:
                if not resp.text.strip():
                    st.error("API returned an empty response.")
                    return None
                return parse_sse_response(resp.text)
            else:
                raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")
        except Exception as e:
            st.error(f"API Request Error: {str(e)}")
            return None

    def summarize_unstructured_answer(answer):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|")\s', answer)
        return "\n".join(f"• {sent.strip()}" for sent in sentences[:6])

    def suggest_sample_questions(query: str) -> List[str]:
        try:
            prompt = (
                f"The user asked: '{query}'. Generate 3-5 clear, concise sample questions related to grants, awards, funding, recipients, or application metrics. "
                f"Format as a numbered list."
            )
            response = complete(st.session_state.model_name, prompt)
            if response:
                questions = []
                for line in response.split("\n"):
                    line = line.strip()
                    if re.match(r'^\d+\.\s*.+', line):
                        question = re.sub(r'^\d+\.\s*', '', line)
                        questions.append(question)
                return questions[:5]
            else:
                return [
                    "What is the total award budget posted by date?",
                    "Which awards have the highest encumbrances?",
                    "What is the total amount of award encumbrances approved?",
                    "What is the date-wise breakdown of award budgets?",
                    "Which awards have pending encumbrances?"
                ]
        except Exception as e:
            return [
                "What is the total award budget posted by date?",
                "Which awards have the highest encumbrances?",
                "What is the total amount of award encumbrances approved?",
                "What is the date-wise breakdown of award budgets?",
                "Which awards have pending encumbrances?"
            ]

    def display_chart_tab(df: pd.DataFrame, prefix: str = "chart", query: str = ""):
        try:
            if df is None or df.empty or len(df.columns) < 2:
                st.warning("No valid data available for visualization.")
                return
            query_lower = query.lower()
            if re.search(r'\b(category|type)\b', query_lower):
                default_data = "Pie Chart"
            elif re.search(r'\b(year|date)\b', query_lower):
                default_data = "Line Chart"
            else:
                default_data = "Bar Chart"
            all_cols = list(df.columns)
            numeric_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(df[col])]
            y_options = numeric_cols + ["All Columns"]
            col1, col2, col3 = st.columns(3)
            x_col = col1.selectbox("X axis", all_cols, index=0, key=f"{prefix}_x")
            remaining_cols = [c for c in y_options if c != x_col]
            y_col = col2.selectbox("Y axis", remaining_cols, index=0, key=f"{prefix}_y")
            chart_options = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart", "Histogram Chart"]
            chart_type = col3.selectbox("Chart Type", chart_options, index=chart_options.index(default_data), key=f"{prefix}_type")
            if y_col == "All Columns":
                df_melted = pd.melt(df, id_vars=[x_col], value_vars=numeric_cols, var_name="Variable", value_name="Value")
                if chart_type == "Line Chart":
                    fig = px.line(df_melted, x=x_col, y="Value", color="Variable", title=chart_type)
                    st.plotly_chart(fig, key=f"{prefix}_line")
                elif chart_type == "Bar Chart":
                    fig = px.bar(df_melted, x=x_col, y="Value", color="Variable", title=chart_type)
                    st.plotly_chart(fig, key=f"{prefix}_bar")
                elif chart_type == "Pie Chart":
                    fig = px.pie(df_melted, names=x_col, values="Value", title=chart_type)
                    st.plotly_chart(fig, key=f"{prefix}_pie")
                elif chart_type == "Scatter Chart":
                    fig = px.scatter(df_melted, x=x_col, y="Value", color="Variable", title=chart_type)
                    st.plotly_chart(fig, key=f"{prefix}_scatter")
                elif chart_type == "Histogram Chart":
                    fig = px.histogram(df_melted, x=x_col, y="Value", color="Variable", title=chart_type)
                    st.plotly_chart(fig, key=f"{prefix}_hist")
            else:
                if chart_type == "Line Chart":
                    fig = px.line(df, x=x_col, y=y_col, title=chart_type)
                    st.plotly_chart(fig, key=f"{prefix}_line")
                elif chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_col, y=y_col, title=chart_type)
                    st.plotly_chart(fig, key=f"{prefix}_bar")
                elif chart_type == "Pie Chart":
                    fig = px.pie(df, names=x_col, values=y_col, title=chart_type)
                    st.plotly_chart(fig, key=f"{prefix}_pie")
                elif chart_type == "Scatter Chart":
                    fig = px.scatter(df, x=x_col, y=y_col, title=chart_type)
                    st.plotly_chart(fig, key=f"{prefix}_scatter")
                elif chart_type == "Histogram Chart":
                    fig = px.histogram(df, x=x_col, y=y_col, title=chart_type)
                    st.plotly_chart(fig, key=f"{prefix}_hist")
        except Exception as e:
            st.error(f"Error generating chart: {str(e)}")

    def toggle_about():
        st.session_state.show_about = not st.session_state.show_about
        st.session_state.show_help = False
        st.session_state.show_history = False

    def toggle_help():
        st.session_state.show_help = not st.session_state.show_help
        st.session_state.show_about = False
        st.session_state.show_history = False

    def toggle_history():
        st.session_state.show_history = not st.session_state.show_history
        st.session_state.show_about = False
        st.session_state.show_help = False

    # --- Sidebar ---
    with st.sidebar:
        logo_url = "https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg"
        st.image(logo_url, width=250)
        if st.button("Clear conversation"):
            start_new_conversation()
        st.radio("Select Data Source:", ["Database", "Document"], key="data_source")
        st.selectbox(
            "Select Cortex Search Service:",
            ["AI.DWH_MART.GRANTS_SEARCH_SERVICES"],
            index=0,
            key="selected_cortex_search_service"
        )
        st.toggle("Debug", key="debug_mode")
        st.toggle("Use chat history", key="use_chat_history")
        with st.expander("Advanced options"):
            st.selectbox("Select model:", MODELS, key="model_name")
            st.number_input(
                "Select number of context chunks",
                value=100,
                key="num_retrieved_chunks",
                min_value=1,
                max_value=400
            )
            st.number_input(
                "Select number of messages to use in chat history",
                value=10,
                key="num_chat_messages",
                min_value=1,
                max_value=100
            )
        if st.button("Sample Questions"):
            st.session_state.show_sample_questions = not st.session_state.get("show_sample_questions", False)
        if st.session_state.get("show_sample_questions", False):
            st.markdown("### Sample Questions")
            sample_questions = [
                "What is the posted budget for awards 41001, 41002, 41003, 41005, 41007, and 41018 by date?",
                "Give me date wise award breakdowns",
                "Give me award breakdowns",
                "Give me date wise award budget, actual award posted, award encumbrance posted, award encumbrance approved",
                "What is the task actual posted by award name?",
                "What is the award budget posted by date for these awards?",
                "What is the total award encumbrance posted for these awards?",
                "What is the total amount of award encumbrances approved?",
                "What is the total actual award posted for these awards?",
                "What is the award budget posted?",
                "What is this document about?",
                "Subject areas",
                "Explain five layers in High level Architecture"
            ]
            for sample in sample_questions:
                if st.button(sample, key=f"sidebar_{sample}"):
                    st.session_state.query = sample
                    st.session_state.show_greeting = False
        with st.expander("Submit Grant Application"):
            st.markdown("### Submit a Grant Application")
            grant_id = st.text_input("Grant ID", key="grant_id")
            applicant_name = st.text_input("Applicant Name", key="applicant_name")
            application_details = st.text_area("Application Details", key="application_details")
            if st.button("Submit Application", key="submit_grant_application"):
                if not grant_id or not applicant_name or not application_details:
                    st.error("Please fill in all fields to submit a grant application.")
                else:
                    success, message = submit_grant_application(grant_id, applicant_name, application_details)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        st.markdown("---")
        if st.button("History"):
            toggle_history()
        if st.session_state.show_history:
            st.markdown("### Recent Questions")
            user_questions = get_user_questions(limit=10)
            if not user_questions:
                st.write("No questions in history yet.")
            else:
                for idx, question in enumerate(user_questions):
                    if st.button(question, key=f"history_{idx}"):
                        st.session_state.query = question
                        st.session_state.show_greeting = False
        if st.button("About"):
            toggle_about()
        if st.session_state.show_about:
            st.markdown("### About")
            st.write(
                "This application uses **Snowflake Cortex Analyst** to interpret "
                "your natural language questions and generate data insights for grants management."
            )
        if st.button("Help & Documentation"):
            toggle_help()
        if st.session_state.show_help:
            st.markdown("### Help & Documentation")
            st.write(
                "- [User Guide](https://docs.snowflake.com/en/guides-overview-ai-features)\n"
                "- [Snowflake Cortex Analyst Docs](https://docs.snowflake.com/)\n"
                "- [Contact Support](https://www.snowflake.com/en/support/)"
            )

    # --- Main UI ---
    st.markdown(
    """
    <div class="fixed-header">
        <h1>Cortex AI – Grants Management Assistant by DiLytics</h1>
        <p><strong>Welcome to Cortex AI. I am here to help with DiLytics Grants Management Insights Solutions.</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)

    semantic_model_filename = SEMANTIC_MODEL.split("/")[-1]
    st.markdown(f"Semantic Model: `{semantic_model_filename}`")
    init_service_metadata()

    if st.session_state.show_greeting and not st.session_state.chat_history:
        st.markdown("Welcome! I’m the Snowflake AI Assistant, ready to assist you with Grants Management. Ask about your grants, funding, recipients, or submit a grant application to get started!")
    else:
        st.session_state.show_greeting = False

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            if message["role"] == "assistant" and "results" in message and message["results"] is not None:
                with st.expander("View SQL Query", expanded=False):
                    st.code(message["sql"], language="sql")
                st.markdown(f"**Query Results ({len(message['results'])} rows):**")
                df_to_display = message["results"].copy()
                # Ensure award numbers are strings and dates are in YYYY-MM-DD format
                award_number_candidates = ["AWARD_NUMBER", "AWARD_ID", "GRANT_ID", "AWARD_NO"]
                for col in award_number_candidates:
                    if col in df_to_display.columns:
                        df_to_display[col] = df_to_display[col].astype(str)
                date_column_candidates = ["DATE", "AWARD_DATE", "SUBMITTED_AT", "DW_DATE_ALLOCATED_KEY"]
                for col in df_to_display.columns:
                    if col in date_column_candidates or "DATE" in col.upper():
                        try:
                            df_to_display[col] = pd.to_datetime(df_to_display[col]).dt.strftime('%Y-%m-%d')
                        except (ValueError, TypeError):
                            continue
                st.dataframe(df_to_display)
                if not df_to_display.empty and len(df_to_display.columns) >= 2:
                    st.markdown("**📈 Visualization:**")
                    # Use message_id to ensure unique keys
                    message_id = message.get("message_id", 0)
                    chart_prefix = f"chart_{message_id}"
                    display_chart_tab(df_to_display, prefix=chart_prefix, query=message.get("query", ""))

    chat_input_query = st.chat_input("Ask your question about grants...")
    if chat_input_query:
        if not st.session_state.authenticated:
            st.warning("Please log in to submit a query.")
        else:
            st.session_state.query = chat_input_query

    if st.session_state.query and st.session_state.authenticated:
        if st.session_state.debug_mode:
            st.sidebar.write(f"Debug: Processing query '{st.session_state.query}'")
            st.sidebar.write(f"Debug: Current message_counter = {st.session_state.message_counter}")
        query = st.session_state.query
        if query.lower().startswith("no of"):
            query = query.replace("no of", "number of", 1)
        st.session_state.show_greeting = False
        st.session_state.chart_x_axis = None
        st.session_state.chart_y_axis = None
        st.session_state.chart_type = "Bar Chart"
        original_query = query
        if query.strip().isdigit() and st.session_state.last_suggestions:
            try:
                index = int(query.strip()) - 1
                if 0 <= index < len(st.session_state.last_suggestions):
                    query = st.session_state.last_suggestions[index]
                else:
                    query = original_query
            except ValueError:
                query = original_query
        is_follow_up = any(re.search(pattern, query.lower()) for pattern in [r'^\bby\b\s+\w+$', r'^\bgroup by\b\s+\w+$']) and st.session_state.previous_query
        combined_query = query
        if st.session_state.use_chat_history and is_follow_up:
            chat_history = get_chat_history()
            if chat_history:
                combined_query = make_chat_history_summary(chat_history, query)
        # Safeguard: Ensure message_counter is initialized
        if "message_counter" not in st.session_state:
            st.session_state.message_counter = 0
            if st.session_state.debug_mode:
                st.sidebar.write("Debug: Re-initialized message_counter to 0 before incrementing")
        # Increment message counter and assign unique message_id
        st.session_state.message_counter += 1
        if st.session_state.debug_mode:
            st.sidebar.write(f"Debug: Incremented message_counter to {st.session_state.message_counter}")
        user_message = {
            "role": "user",
            "content": original_query,
            "message_id": st.session_state.message_counter
        }
        st.session_state.chat_history.append(user_message)
        st.session_state.messages.append({"role": "user", "content": original_query})
        with st.chat_message("user"):
            st.markdown(original_query)
        with st.chat_message("assistant"):
            with st.spinner("Generating Response..."):
                response_placeholder = st.empty()
                if st.session_state.data_source not in ["Database", "Document"]:
                    st.session_state.data_source = "Database"
                is_structured = is_structured_query(combined_query) and st.session_state.data_source == "Database"
                is_complete = is_complete_query(combined_query)
                is_summarize = is_summarize_query(combined_query)
                is_suggestion = is_question_suggestion_query(combined_query)
                is_greeting = is_greeting_query(combined_query)
                # Increment message counter for assistant response
                if "message_counter" not in st.session_state:
                    st.session_state.message_counter = 0
                    if st.session_state.debug_mode:
                        st.sidebar.write("Debug: Re-initialized message_counter to 0 before assistant increment")
                st.session_state.message_counter += 1
                if st.session_state.debug_mode:
                    st.sidebar.write(f"Debug: Incremented message_counter to {st.session_state.message_counter} for assistant")
                assistant_response = {
                    "role": "assistant",
                    "content": "",
                    "query": combined_query,
                    "message_id": st.session_state.message_counter
                }
                response_content = ""
                failed_response = False

                if is_greeting:
                    response_content = (
                        "Hello! How can I assist you with Grants Management today?\n"
                        "Here are some questions you can try:\n"
                        "1. What is the total award budget posted by date?\n"
                        "2. Which awards have the highest encumbrances?\n"
                        "3. What is the total amount of award encumbrances approved?\n"
                        "4. What is the date-wise breakdown of award budgets?\n"
                        "5. Which awards have pending encumbrances?\n"
                    )
                    response_placeholder.markdown(response_content, unsafe_allow_html=True)
                    assistant_response["content"] = response_content
                    suggestions = [
                        "What is the total award budget posted by date?",
                        "Which awards have the highest encumbrances?",
                        "What is the total amount of award encumbrances approved?",
                        "What is the date-wise breakdown of award budgets?",
                        "Which awards have pending encumbrances?"
                    ]
                    st.session_state.last_suggestions = suggestions
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

                elif is_suggestion:
                    response_content = "Here are some questions you can try:\n"
                    suggestions = [
                        "What is the total award budget posted by date?",
                        "Which awards have the highest encumbrances?",
                        "What is the total amount of award encumbrances approved?",
                        "What is the date-wise breakdown of award budgets?",
                        "Which awards have pending encumbrances?"
                    ]
                    for i, suggestion in enumerate(suggestions, 1):
                        response_content += f"{i}. {suggestion}\n"
                    response_placeholder.markdown(response_content, unsafe_allow_html=True)
                    assistant_response["content"] = response_content
                    st.session_state.last_suggestions = suggestions
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

                elif is_complete:
                    response = create_prompt(combined_query)
                    if response:
                        response_content = f"**Generated Response:**\n{response}"
                        response_placeholder.markdown(response_content, unsafe_allow_html=True)
                        assistant_response["content"] = response_content
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                    else:
                        failed_response = True

                elif is_summarize:
                    summary = summarize(combined_query)
                    if summary:
                        response_content = f"**Summary:**\n{summary}"
                        response_placeholder.markdown(response_content, unsafe_allow_html=True)
                        assistant_response["content"] = response_content
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                    else:
                        failed_response = True

                elif is_structured:
                    response = snowflake_api_call(combined_query, is_structured=True)
                    sql, _ = process_sse_response(response, is_structured=True)
                    if sql:
                        results = run_snowflake_query(sql)
                        if results is not None and not results.empty:
                            results_text = results.to_string(index=False)
                            prompt = f"Provide a concise natural language answer to the query '{combined_query}' using the following data:\n\n{results_text}"
                            summary = complete(st.session_state.model_name, prompt)
                            if not summary:
                                summary = "Unable to generate a summary."
                            response_content = f"**Generated Response:**\n{summary}"
                            response_placeholder.markdown(response_content, unsafe_allow_html=True)
                            with st.expander("View SQL Query", expanded=False):
                                st.code(sql, language="sql")
                            st.markdown(f"**Query Results ({len(results)} rows):**")
                            df_to_display = results.copy()
                            # Ensure award numbers are strings and dates are in YYYY-MM-DD format
                            award_number_candidates = ["AWARD_NUMBER", "AWARD_ID", "GRANT_ID", "AWARD_NO"]
                            for col in award_number_candidates:
                                if col in df_to_display.columns:
                                    df_to_display[col] = df_to_display[col].astype(str)
                            date_column_candidates = ["DATE", "AWARD_DATE", "SUBMITTED_AT", "DW_DATE_ALLOCATED_KEY"]
                            for col in df_to_display.columns:
                                if col in date_column_candidates or "DATE" in col.upper():
                                    try:
                                        df_to_display[col] = pd.to_datetime(df_to_display[col]).dt.strftime('%Y-%m-%d')
                                    except (ValueError, TypeError):
                                        continue
                            st.dataframe(df_to_display)
                            if len(df_to_display.columns) >= 2:
                                st.markdown("**📈 Visualization:**")
                                # Use assistant's message_id for chart prefix
                                chart_prefix = f"chart_{assistant_response['message_id']}"
                                display_chart_tab(df_to_display, prefix=chart_prefix, query=combined_query)
                            assistant_response.update({
                                "content": response_content,
                                "sql": sql,
                                "results": results,
                                "summary": summary
                            })
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response_content,
                                "sql": sql,
                                "results": results,
                                "summary": summary
                            })
                        else:
                            response_content = "No data returned for the query."
                            failed_response = True
                            assistant_response["content"] = response_content
                    else:
                        response_content = "Failed to generate SQL query."
                        failed_response = True
                        assistant_response["content"] = response_content

                elif st.session_state.data_source == "Document":
                    response = snowflake_api_call(combined_query, is_structured=False)
                    _, search_results = process_sse_response(response, is_structured=False)
                    if search_results:
                        raw_result = search_results[0]
                        summary = create_prompt(combined_query)
                        if summary:
                            response_content = f"**Answer:**\n{summary}"
                        else:
                            response_content = f"**Key Information:**\n{summarize_unstructured_answer(raw_result)}"
                        response_placeholder.markdown(response_content, unsafe_allow_html=True)
                        assistant_response["content"] = response_content
                        st.session_state.messages.append({"role": "assistant", "content": response_content})
                    else:
                        failed_response = True

                if failed_response:
                    suggestions = suggest_sample_questions(combined_query)
                    response_content = "I am not sure about your question. Here are some questions you can ask me:\n"
                    for i, suggestion in enumerate(suggestions, 1):
                        response_content += f"{i}. {suggestion}\n"
                    response_placeholder.markdown(response_content, unsafe_allow_html=True)
                    assistant_response["content"] = response_content
                    st.session_state.last_suggestions = suggestions
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

                st.session_state.chat_history.append(assistant_response)
                st.session_state.current_query = combined_query
                st.session_state.current_results = assistant_response.get("results")
                st.session_state.current_sql = assistant_response.get("sql")
                st.session_state.current_summary = assistant_response.get("summary")
                st.session_state.previous_query = combined_query
                st.session_state.previous_sql = assistant_response.get("sql")
                st.session_state.previous_results = assistant_response.get("results")
                st.session_state.query = None
