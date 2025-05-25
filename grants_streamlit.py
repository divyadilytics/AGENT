import streamlit as st
import json
import re
import requests
import snowflake.connector
import pandas as pd
from snowflake.snowpark import Session
from typing import Any, Dict, List, Optional, Tuple
import plotly.express as px  # Added for interactive visualizations
from collections import Counter # For summarizing unstructured text

# Snowflake/Cortex Configuration
HOST = "bnkzyio-ljb86662.snowflakecomputing.com" # Ensure this is your Snowflake account URL
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000  # in milliseconds

# Single semantic model & search service
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/GRANTSyaml.yaml'
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.TRAIL_SEARCH_SERVICES"

# Streamlit Page Config
st.set_page_config(
    page_title="❄️ Cortex AI Assistant for Grants",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize session state variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.CONN = None
    st.session_state.snowpark_session = None

if 'messages' not in st.session_state:
    st.session_state.messages = [] # Replaces chat_history for structured messages
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'current_query_to_process' not in st.session_state:
    st.session_state.current_query_to_process = None
if 'show_suggested_buttons' not in st.session_state:
    st.session_state.show_suggested_buttons = False
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'selected_history_query' not in st.session_state:
    st.session_state.selected_history_query = None
if 'query_results' not in st.session_state:
    st.session_state.query_results = {} # Maps question to response data (DataFrame or text)

# Initialize chart selection persistence from Code 1
if "chart_x_axis" not in st.session_state:
    st.session_state.chart_x_axis = None
if "chart_y_axis" not in st.session_state:
    st.session_state.chart_y_axis = None
if "chart_type" not in st.session_state:
    st.session_state.chart_type = "Bar Chart"

# Hide Streamlit branding and prevent chat history shading
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
/* Prevent shading of previous chat messages */
[data-testid="stChatMessage"] {
    opacity: 1 !important;
    background-color: transparent !important;
}
.stButton button[kind="primary"] {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
    background-color: #29B5E8; /* Snowflake blue */
    color: white;
    border-radius: 5px;
    padding: 8px 16px;
    border: none;
}
.stExpander {
    border: none;
    background-color: transparent;
}
.stExpander > div > div > button::before {
    content: "▶ ";
    color: #29B5E8; /* Snowflake blue */
}
.stExpander[open] > div > div > button::before {
    content: "▼ ";
    color: #29B5E8; /* Snowflake blue */
}
.search-button button, .quit-button button {
    padding: 0px;
    min-height: 20px;
    height: 20px;
    width: 20px;
    font-size: 16px;
    line-height: 20px;
    background-color: transparent;
    color: #000000;
    border: none;
    vertical-align: middle;
}
.search-button button:hover, .quit-button button:hover {
    background-color: transparent;
    color: #29B5E8; /* Snowflake blue */
}
</style>
""", unsafe_allow_html=True)

# Function to start a new conversation
def start_new_conversation():
    st.session_state.messages = []
    st.session_state.current_query_to_process = None
    st.session_state.show_suggested_buttons = False
    st.session_state.show_history = False
    st.session_state.selected_history_query = None
    st.session_state.query_results = {}
    st.session_state.chart_x_axis = None
    st.session_state.chart_y_axis = None
    st.session_state.chart_type = "Bar Chart"
    st.rerun()

# Authentication logic (from Code 1, adapted for clarity)
if not st.session_state.authenticated:
    st.title("Welcome to Snowflake Cortex AI Assistant")
    st.markdown("Please login to interact with your data.")

    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")

    if st.button("Login"):
        try:
            conn = snowflake.connector.connect(
                user=st.session_state.username,
                password=st.session_state.password,
                account=HOST.split('.')[0], # Extract account from host
                host=HOST,
                port=443,
                warehouse="COMPUTE_WH", # Ensure you have access to this warehouse
                role="ACCOUNTADMIN", # Or your preferred role
                database=DATABASE,
                schema=SCHEMA,
            )
            st.session_state.CONN = conn

            # Using Snowpark Session for convenience with .sql() and .to_pandas()
            snowpark_session = Session.builder.configs({
                "connection": conn
            }).create()
            st.session_state.snowpark_session = snowpark_session

            # Set session context (optional, but good practice)
            with conn.cursor() as cur:
                cur.execute(f"USE DATABASE {DATABASE}")
                cur.execute(f"USE SCHEMA {SCHEMA}")
                cur.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = TRUE")

            st.session_state.authenticated = True
            st.success("Authentication successful! Redirecting...")
            st.rerun()

        except Exception as e:
            st.error(f"Authentication failed: {e}")
else:
    # Ensure session is available if already authenticated
    session = st.session_state.snowpark_session

    # --- Utility Functions (Combined & Refined) ---
    def run_snowflake_query(query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Executes a SQL query against Snowflake and returns results as a Pandas DataFrame."""
        try:
            if not query:
                return None, "⚠️ No SQL query generated."
            df = session.sql(query)
            pandas_df = df.to_pandas()
            return pandas_df, None
        except Exception as e:
            return None, f"❌ SQL Execution Error: {str(e)}"

    def is_structured_query(query: str) -> bool:
        """
        Determines if a query is structured based on keywords typically associated with data queries.
        (From Code 2, more comprehensive)
        """
        structured_keywords = [
            "total", "show", "top", "funding", "net increase", "net decrease", "group by", "order by",
            "how much", "give", "count", "avg", "max", "min", "least", "highest", "by year",
            "how many", "total amount", "version", "scenario", "forecast", "year", "savings",
            "award", "position", "budget", "allocation", "expenditure", "department", "variance",
            "breakdown", "comparison", "change", "completed units", "jurisdiction", "month", "date"
        ]
        unstructured_keywords = [
            "describe", "introduction", "summary", "tell me about", "overview", "explain", "what is this document about"
        ]
        query_lower = query.lower()
        # If it explicitly asks for unstructured content, generally lean that way unless heavily structured
        if any(keyword in query_lower for keyword in unstructured_keywords):
            structured_score = sum(1 for keyword in structured_keywords if keyword in query_lower)
            if structured_score < 2: # Threshold to override unstructured intent
                return False
        return any(keyword in query_lower for keyword in structured_keywords)

    def is_unstructured_query(query: str) -> bool:
        """Detects if a query is asking for unstructured information."""
        unstructured_keywords = [
            "policy", "document", "description", "summary", "highlight", "explain", "describe", "guidelines",
            "procedure", "how to", "define", "definition", "rules", "steps", "overview",
            "objective", "purpose", "benefits", "importance", "impact", "details", "regulation",
            "requirement", "compliance", "when to", "where to", "meaning", "interpretation",
            "clarify", "note", "explanation", "instructions", "what is this document about", "list all subject areas"
        ]
        query_lower = query.lower()
        return any(word in query_lower for word in unstructured_keywords)

    def detect_yaml_or_sql_intent(query: str) -> bool:
        """Detects if a query is asking for information about the semantic model (YAML) or SQL structure."""
        yaml_keywords = ["yaml", "semantic model", "metric", "dimension", "table", "column", "sql for", "structure of"]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in yaml_keywords)

    def preprocess_query(query: str) -> List[str]:
        """Extracts key terms from the query to improve search relevance."""
        query_lower = query.lower()
        tokens = re.findall(r'\b\w+\b', query_lower) # More robust tokenization
        stopwords = set(['what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to', 'how', 'many', 'much', 'me', 'give', 'show', 'by', 'this', 'document', 'about', 'all', 'list', 'areas', 'subject'])
        key_terms = [token for token in tokens if token not in stopwords and token.isalnum()]
        def normalize_term(term):
            return re.sub(r'(ing|s|ed|ly)$', '', term) # More comprehensive stemming
        return [normalize_term(term) for term in key_terms]

    def parse_sse_response(response_text: str) -> List[Dict]:
        """Parse SSE response text into a list of JSON objects. (From Code 1)"""
        events = []
        lines = response_text.strip().split("\n")
        current_event = {}
        for line in lines:
            if line.startswith("event:"):
                current_event["event"] = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data_str = line.split(":", 1)[1].strip()
                if data_str != "[DONE]":  # Skip the [DONE] marker
                    try:
                        data_json = json.loads(data_str)
                        current_event["data"] = data_json
                        events.append(current_event)
                        current_event = {}  # Reset for next event
                    except json.JSONDecodeError as e:
                        if st.session_state.debug_mode:
                            st.error(f"❌ Failed to parse SSE data: {str(e)} - Data: {data_str}")
        return events

    def snowflake_api_call(query: str, is_structured: bool = False, is_yaml: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Makes an API call to Snowflake Cortex, routing to text-to-SQL or search service
        based on the query type. Uses requests.post for cloud deployment.
        """
        payload = {
            "model": "llama3.1-70b", # Using a powerful model (Code 2 had llama3.1-70b)
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "tools": []
        }
        if is_structured or is_yaml:
            payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
            payload["tool_resources"] = {"analyst1": {"semantic_model_file": SEMANTIC_MODEL}}
        else:
            payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
            payload["tool_resources"] = {"search1": {"name": CORTEX_SEARCH_SERVICES, "max_results": 10}}

        try:
            # Get the Snowflake token from the active connection
            # This relies on st.session_state.CONN being a valid snowflake.connector.Connection object
            if st.session_state.CONN is None:
                raise Exception("Snowflake connection is not established.")

            token = st.session_state.CONN.rest.token
            if not token:
                raise Exception("Snowflake token not found.")

            resp = requests.post(
                url=f"https://{HOST}{API_ENDPOINT}",
                json=payload,
                headers={
                    "Authorization": f'Snowflake Token="{token}"',
                    "Content-Type": "application/json",
                },
                timeout=API_TIMEOUT // 1000 # requests timeout is in seconds
            )
            if st.session_state.debug_mode:
                st.write(f"Debug: API Request Payload for query '{query}': {payload}")
                st.write(f"Debug: Raw API Response for query '{query}': {resp.text}")

            if resp.status_code < 400:
                if not resp.text.strip():
                    return None, "API returned an empty response."
                # Parse SSE response text into a list of events
                return resp.text, None
            else:
                return None, f"Failed request with status {resp.status_code}: {resp.text}"
        except Exception as e:
            return None, f"❌ API Request Failed: {str(e)}"

    def process_sse_response(response_text: str, is_structured: bool, query: str) -> Tuple[str, str, List[str], Optional[str]]:
        """
        Processes the SSE response from Snowflake Cortex, extracting SQL/explanation
        for structured queries or search results for unstructured queries.
        (Combines Code 1's parse_sse_response call with Code 2's processing logic)
        """
        sql = ""
        explanation = ""
        search_results = []
        error = None

        events = parse_sse_response(response_text)
        if not events:
            return sql, explanation, search_results, "No events parsed from API response."

        try:
            for event in events:
                if event.get("event") == "message.delta" and "data" in event:
                    delta = event["data"].get("delta", {})
                    content = delta.get("content", [])
                    for item in content:
                        if item.get("type") == "tool_results":
                            tool_results = item.get("tool_results", {})
                            if 'content' in tool_results:
                                for result in tool_results['content']:
                                    if result.get('type') == 'json':
                                        result_data = result.get('json', {})
                                        if is_structured:
                                            if 'sql' in result_data:
                                                sql = result_data.get('sql', '')
                                            if 'explanation' in result_data:
                                                explanation = result_data.get('explanation', '')
                                        else:
                                            if 'searchResults' in result_data:
                                                key_terms = preprocess_query(query)
                                                ranked_results = []
                                                for sr in result_data['searchResults']:
                                                    text = sr["text"]
                                                    text_lower = text.lower()
                                                    score = sum(1 for term in key_terms if term in text_lower)
                                                    ranked_results.append((text, score))
                                                ranked_results.sort(key=lambda x: x[1], reverse=True)
                                                # Summarize each top result individually, then filter
                                                summarized_parts = []
                                                for text, _ in ranked_results[:5]: # Take top 5 results for summarizing
                                                    summary_part = summarize_text_with_cortex_or_local(text, query)
                                                    if summary_part and "No relevant content found" not in summary_part:
                                                        summarized_parts.append(summary_part)
                                                search_results = summarized_parts
            if not is_structured and not sql and not explanation and not search_results:
                error = "Cortex Analyst/Search did not return a valid response (no SQL, explanation, or search results)."
        except Exception as e:
            error = f"❌ Error Processing SSE Response: {str(e)}"
        if st.session_state.debug_mode:
            st.write(f"Debug: Processed SSE Response - SQL: {sql}, Explanation: {explanation}, Search Results: {search_results}, Error: {error}")
        return sql.strip(), explanation.strip(), search_results, error

    def summarize_text_with_cortex_or_local(text: str, query: str) -> str:
        """Calls Snowflake Cortex SUMMARIZE function with cleaned input text, with local fallback."""
        try:
            # Clean text for Snowflake UDF
            text = re.sub(r'\s+', ' ', text.strip())
            text = text.replace("'", "''") # Escape single quotes for SQL
            query_sql = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
            result = session.sql(query_sql).collect()
            summary = result[0]["SUMMARY"]
            if summary and len(summary.strip()) > 50: # Ensure substantial summary
                return summary
            raise Exception("Cortex SUMMARIZE returned empty or too short summary.")
        except Exception as e:
            if st.session_state.debug_mode:
                st.write(f"Debug: Cortex SUMMARIZE failed or returned short summary: {str(e)}. Using local fallback.")
            return summarize_unstructured_answer(text, query) # Local fallback

    def summarize_unstructured_answer(answer: str, query: str) -> str:
        """Summarizes unstructured text by ranking sentences based on query relevance with weighted scoring."""
        # Clean specific intros if present, like "Program Overview"
        answer = re.sub(r"^.*?Program\sOverview", "Program Overview", answer, flags=re.DOTALL)
        sentences = re.split(r'(?<=\.|\?|\!)\s+', answer)
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        if not sentences:
            return "No relevant content found."

        key_terms = preprocess_query(query)
        
        # Calculate term frequency in the whole answer to penalize common words
        all_words = ' '.join(sentences).lower().split()
        word_counts = Counter(all_words)
        total_words = len(all_words) + 1 # Avoid division by zero

        # Give higher weight to query terms that are less common in the overall text
        term_weights = {term: max(1.0, 3.0 * (1 - word_counts.get(term.lower(), 0) / total_words)) for term in key_terms}

        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(term_weights[term] for term in key_terms if term in sent_lower)
            scored_sentences.append((sent, score))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        # Select top sentences, ensuring they have some relevance score
        top_sentences = [sent for sent, score in scored_sentences[:5] if score > 0]
        
        # Fallback if no relevant sentences found (e.g., query terms not in text)
        if not top_sentences and sentences:
            top_sentences = sentences[:min(len(sentences), 5)] # Take up to 5 sentences

        return "\n\n".join(f"• {sent}" for sent in top_sentences) if top_sentences else "No relevant content found."

    def format_results_for_history(df: pd.DataFrame) -> str:
        """Formats a Pandas DataFrame into a Markdown table for chat history."""
        if df is None or df.empty:
            return "No data found."
        if len(df.columns) == 1 and len(df) == 1: # Single value case
            return str(df.iloc[0, 0])
        return df.to_markdown(index=False)

    def process_followup_query(followup_query: str, parent_query: str):
        """
        Processes a follow-up query using only the parent query's response data.
        (From Code 2, adapted for combined structure)
        """
        if parent_query not in st.session_state.query_results:
            return f"⚠️ No response data available for parent query: `{parent_query}`. Please start a new query."
        
        response_data = st.session_state.query_results[parent_query]
        response_content = ""
        
        if isinstance(response_data, pd.DataFrame):
            # Structured query follow-up
            df = response_data
            followup_lower = followup_query.lower()
            
            key_terms = preprocess_query(followup_query)
            award_number = None
            # Look for award numbers or IDs that are numeric or alphanumeric strings
            potential_ids = re.findall(r'\b\d{4,}\w*\b|\b\w*\d{4,}\b', followup_lower)
            if potential_ids:
                award_number = potential_ids[0]
            
            # Find columns based on query terms or common patterns
            requested_column = None
            for term in key_terms:
                for col in df.columns:
                    if term in col.lower() or term.rstrip('s') in col.lower(): # account for plurals
                        requested_column = col
                        break
                if requested_column:
                    break
            
            # Find award number column
            award_number_col = None
            for col in df.columns:
                if 'award' in col.lower() and ('number' in col.lower() or 'id' in col.lower() or 'no' in col.lower() or 'name' in col.lower()):
                    award_number_col = col
                    break
            
            if requested_column:
                if award_number and award_number_col and award_number_col in df.columns:
                    # Filter by award number (case-insensitive, partial match)
                    # Convert to string and handle potential "l" vs "1", "o" vs "0" for robustness
                    award_number_str = str(award_number).lower().replace('l', '1').replace('o', '0')
                    result = df[df[award_number_col].astype(str).str.lower().str.contains(award_number_str, na=False)]
                    
                    if not result.empty:
                        # If multiple results, try to pick the first relevant one or list them
                        values = result[requested_column].tolist()
                        if len(values) == 1:
                            value_str = str(values[0])
                        else:
                            value_str = ", ".join(map(str, values))
                        response_content = f"**{requested_column}** for award matching '{award_number}': {value_str}"
                    else:
                        available_awards = df[award_number_col].astype(str).unique() if award_number_col else []
                        response_content = f"No data found for award matching '{award_number}' in the parent response. Available awards: {', '.join(available_awards[:5])}..." # Limit list
                else:
                    # Return all values for the requested column if no specific award is identified or award column is missing
                    if requested_column in df.columns:
                        response_content = f"**{requested_column}** values:\n{df[requested_column].to_markdown(index=False)}"
                    else:
                        response_content = f"The column '{requested_column}' was not found in the previous results. Available columns: {', '.join(df.columns)}"
            else:
                available_columns = ', '.join(df.columns)
                response_content = f"No relevant column or specific filter found for '{followup_query}' in the parent response. Available columns: {available_columns}"
        
        else: # Unstructured query follow-up (response_data is string)
            parent_text = response_data
            response_content = summarize_text_with_cortex_or_local(parent_text, followup_query)
            if response_content == "No relevant content found.":
                response_content = f"No relevant information found for '{followup_query}' in the parent response."
        
        if st.session_state.debug_mode:
            st.write(f"Debug: Follow-up query '{followup_query}' for parent '{parent_query}' - Response: {response_content}")
        
        return response_content

    # Visualization Function (from Code 1)
    def display_chart_tab(df: pd.DataFrame, prefix: str = "chart", query: str = ""):
        """Allows user to select chart options and displays a chart with unique widget keys."""
        if df.empty or len(df.columns) < 2:
            st.warning("Not enough data or columns to generate a chart.")
            return

        # Determine default chart type based on query
        query_lower = query.lower()
        default_chart = "Bar Chart" # Default fallback
        if re.search(r'\b(county|jurisdiction|department)\b', query_lower) and any(col.lower() in query_lower for col in ['count', 'total', 'sum']):
            default_chart = "Pie Chart"
        elif re.search(r'\b(month|year|date)\b', query_lower):
            default_chart = "Line Chart"

        all_cols = list(df.columns)
        
        # Ensure column types are numeric for Y-axis if suitable
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()

        col1, col2, col3 = st.columns(3)

        # X-axis: prefer non-numeric for categories, or first available
        default_x_options = non_numeric_cols + numeric_cols
        default_x_val = st.session_state.get(f"{prefix}_x", default_x_options[0] if default_x_options else all_cols[0])
        try:
            x_index = all_cols.index(default_x_val)
        except ValueError:
            x_index = 0
        x_col = col1.selectbox("X axis", all_cols, index=x_index, key=f"{prefix}_x_select")

        # Y-axis: prefer numeric for values
        remaining_numeric_cols = [c for c in numeric_cols if c != x_col]
        default_y_options = remaining_numeric_cols + [c for c in all_cols if c not in numeric_cols and c != x_col]
        default_y_val = st.session_state.get(f"{prefix}_y", default_y_options[0] if default_y_options else (all_cols[1] if len(all_cols) > 1 else all_cols[0]))
        try:
            y_index = all_cols.index(default_y_val)
        except ValueError:
            y_index = 0
        y_col = col2.selectbox("Y axis", all_cols, index=y_index, key=f"{prefix}_y_select")

        chart_options = ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Chart", "Histogram Chart"]
        default_type = st.session_state.get(f"{prefix}_type", default_chart)
        try:
            type_index = chart_options.index(default_type)
        except ValueError:
            type_index = chart_options.index("Bar Chart") # Fallback
        chart_type = col3.selectbox("Chart Type", chart_options, index=type_index, key=f"{prefix}_type_select")

        # Ensure Y-axis is numeric for charts that require it
        if chart_type in ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Chart"] and df[y_col].dtype not in ['int64', 'float64']:
            st.warning(f"Warning: Y-axis column '{y_col}' is not numeric. Chart might not render correctly.")
            return

        try:
            if chart_type == "Line Chart":
                fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            elif chart_type == "Bar Chart":
                fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            elif chart_type == "Pie Chart":
                fig = px.pie(df, names=x_col, values=y_col, title=f"Distribution of {y_col} by {x_col}")
            elif chart_type == "Scatter Chart":
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
            elif chart_type == "Histogram Chart":
                fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col}")
            st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_plot")
        except Exception as e:
            st.error(f"Failed to generate chart: {e}")

    # Core Query Processing and Display Function (Combines main logic from both apps)
    def process_query_and_display(query: str, is_followup: bool = False, parent_query: str = None):
        """
        Processes a user query or follow-up, interacts with Cortex for new queries,
        and updates session state.
        """
        st.session_state.show_suggested_buttons = False
        
        if is_followup and parent_query:
            # Handle follow-up query
            st.session_state.messages.append({"role": "user", "content": query, "parent_query": parent_query})
            with st.chat_message("user"):
                st.markdown(f"Follow-up: {query}")
            
            with st.chat_message("assistant"):
                response_content = process_followup_query(query, parent_query)
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content, "parent_query": parent_query})
        else:
            # Handle new query
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)
            
            response_content_for_history = ""
            response_data = None
            sql_generated = None
            
            with st.chat_message("assistant"):
                with st.spinner("Thinking... 🤖"):
                    is_structured = is_structured_query(query)
                    is_unstructured = is_unstructured_query(query)
                    is_yaml_or_sql_intent = detect_yaml_or_sql_intent(query)
                    
                    if is_structured or is_yaml_or_sql_intent:
                        response_text_from_api, api_error = snowflake_api_call(query, is_structured=True, is_yaml=is_yaml_or_sql_intent)
                        
                        if api_error:
                            st.error(api_error)
                            response_content_for_history = api_error
                            st.session_state.show_suggested_buttons = True
                        else:
                            final_sql, explanation, _, sse_error = process_sse_response(response_text_from_api, is_structured=True, query=query)
                            sql_generated = final_sql # Store for history display
                            if sse_error:
                                st.error(sse_error)
                                response_content_for_history = sse_error
                                st.session_state.show_suggested_buttons = True
                            elif final_sql:
                                st.markdown("**📜 SQL Query:**")
                                st.code(final_sql, language='sql')
                                response_content_for_history += f"**📜 SQL Query:**\n```sql\n{final_sql}\n```\n"
                                if explanation:
                                    st.markdown("**📘 Explanation:**")
                                    st.write(explanation)
                                    response_content_for_history += f"**📘 Explanation:**\n{explanation}\n"
                                
                                results_df, query_error = run_snowflake_query(final_sql)
                                if query_error:
                                    st.error(query_error)
                                    response_content_for_history += query_error
                                    st.session_state.show_suggested_buttons = True
                                elif results_df is not None and not results_df.empty:
                                    st.markdown("**📊 Results:**")
                                    if len(results_df.columns) == 1 and len(results_df) == 1:
                                        st.write(f"**{results_df.iloc[0, 0]}**")
                                    else:
                                        st.dataframe(results_df)
                                    response_content_for_history += "**📊 Results:**\n" + format_results_for_history(results_df)
                                    response_data = results_df # Store DataFrame for follow-ups

                                    # Display visualization from Code 1
                                    if not results_df.empty and len(results_df.columns) >= 2:
                                        st.markdown("**📈 Visualization:**")
                                        # Use a unique key for the chart container to ensure re-rendering
                                        with st.container(key=f"chart_container_{hash(query)}"):
                                            display_chart_tab(results_df, prefix=f"chart_{hash(query)}", query=query)
                                else:
                                    st.markdown("⚠️ No data found for the generated SQL query.")
                                    response_content_for_history += "⚠️ No data found for the generated SQL query.\n"
                                    st.session_state.show_suggested_buttons = True
                            else:
                                st.markdown("⚠️ No SQL generated. Could not understand the structured/YAML query.")
                                response_content_for_history = "⚠️ No SQL generated. Could not understand the structured/YAML query.\n"
                                st.session_state.show_suggested_buttons = True
                    
                    elif is_unstructured:
                        response_text_from_api, api_error = snowflake_api_call(query, is_structured=False)
                        if api_error:
                            st.error(api_error)
                            response_content_for_history = api_error
                            st.session_state.show_suggested_buttons = True
                        else:
                            _, _, search_results, sse_error = process_sse_response(response_text_from_api, is_structured=False, query=query)
                            if sse_error:
                                st.error(sse_error)
                                response_content_for_history = sse_error
                                st.session_state.show_suggested_buttons = True
                            elif search_results:
                                st.markdown("**🔍 Document Highlights:**")
                                combined_summaries = "\n\n".join(search_results)
                                st.write(combined_summaries)
                                response_content_for_history += f"**🔍 Document Highlights:**\n{combined_summaries}\n"
                                response_data = combined_summaries # Store text for follow-ups
                            else:
                                st.markdown(f"### I couldn't find information for: '{query}'")
                                st.markdown("Try rephrasing your question or selecting from the suggested questions below.")
                                response_content_for_history = f"### I couldn't find information for: '{query}'\nTry rephrasing your question or selecting from the suggested questions."
                                st.session_state.show_suggested_buttons = True
                    else: # General case, try unstructured first if no clear structure
                        response_text_from_api, api_error = snowflake_api_call(query, is_structured=False)
                        if api_error:
                            st.error(api_error)
                            response_content_for_history = api_error
                            st.session_state.show_suggested_buttons = True
                        else:
                            _, _, search_results, sse_error = process_sse_response(response_text_from_api, is_structured=False, query=query)
                            if sse_error or not search_results: # If unstructured fails or empty, try structured as a fallback
                                if st.session_state.debug_mode:
                                    st.write(f"Debug: Unstructured failed or empty, trying structured as fallback for query: {query}")
                                response_text_from_api, api_error = snowflake_api_call(query, is_structured=True)
                                if api_error:
                                    st.error(api_error)
                                    response_content_for_history = api_error
                                    st.session_state.show_suggested_buttons = True
                                else:
                                    final_sql, explanation, _, sse_error = process_sse_response(response_text_from_api, is_structured=True, query=query)
                                    sql_generated = final_sql
                                    if final_sql:
                                        st.markdown("**📜 SQL Query:**")
                                        st.code(final_sql, language='sql')
                                        response_content_for_history += f"**📜 SQL Query:**\n```sql\n{final_sql}\n```\n"
                                        if explanation:
                                            st.markdown("**📘 Explanation:**")
                                            st.write(explanation)
                                            response_content_for_history += f"**📘 Explanation:**\n{explanation}\n"
                                        results_df, query_error = run_snowflake_query(final_sql)
                                        if query_error:
                                            st.error(query_error)
                                            response_content_for_history += query_error
                                            st.session_state.show_suggested_buttons = True
                                        elif results_df is not None and not results_df.empty:
                                            st.markdown("**📊 Results:**")
                                            if len(results_df.columns) == 1 and len(results_df) == 1:
                                                st.write(f"**{results_df.iloc[0, 0]}**")
                                            else:
                                                st.dataframe(results_df)
                                            response_content_for_history += "**📊 Results:**\n" + format_results_for_history(results_df)
                                            response_data = results_df
                                            if not results_df.empty and len(results_df.columns) >= 2:
                                                st.markdown("**📈 Visualization:**")
                                                with st.container(key=f"chart_container_{hash(query)}"):
                                                    display_chart_tab(results_df, prefix=f"chart_{hash(query)}", query=query)
                                        else:
                                            st.markdown("⚠️ No data found for the generated SQL query.")
                                            response_content_for_history += "⚠️ No data found for the generated SQL query.\n"
                                            st.session_state.show_suggested_buttons = True
                                    else:
                                        st.markdown(f"### I couldn't find information for: '{query}'")
                                        st.markdown("Try rephrasing your question or selecting from the suggested questions below.")
                                        response_content_for_history = f"### I couldn't find information for: '{query}'\nTry rephrasing your question or selecting from the suggested questions."
                                        st.session_state.show_suggested_buttons = True
                            else: # Unstructured query succeeded
                                st.markdown("**🔍 Document Highlights:**")
                                combined_summaries = "\n\n".join(search_results)
                                st.write(combined_summaries)
                                response_content_for_history += f"**🔍 Document Highlights:**\n{combined_summaries}\n"
                                response_data = combined_summaries
                    
                    st.session_state.messages.append({"role": "assistant", "content": response_content_for_history, "sql": sql_generated if sql_generated else ""})
                    if response_data is not None:
                        st.session_state.query_results[query] = response_data

    # Suggested questions
    suggested_questions = [
        "What is the posted budget for awards 41001, 41002, 41003, 41005, 41007, and 41018 by date?",
        "Give me date wise award breakdowns",
        "Give me award breakdowns",
        "Give me date wise award budget, actual award posted,award encunbrance posted,award encumbrance approved",
        "What is the task actual posted by award name?",
        "What is the award budget posted by date for these awards?",
        "What is the total award encumbrance posted for these awards?",
        "What is the total amount of award encumbrances approved?",
        "What is the total actual award posted for these awards?",
        "what is the award budget posted?",
        "what is this document about",
        "list all subject areas"
    ]

    def main():
        # History button in top right corner
        col1, col2 = st.columns([9, 1])
        with col2:
            if st.button("📜 Toggle History", key="history_toggle", type="primary"):
                st.session_state.show_history = not st.session_state.show_history
                st.session_state.selected_history_query = None  # Reset selected query

        # Sidebar setup
        with st.sidebar:
            st.markdown("""
            <style>
            [data-testid="stSidebar"] [data-testid="stButton"] > button {
                background-color: #29B5E8 !important;
                color: white !important;
                font-weight: bold !important;
                width: 100% !important;
                border-radius: 0px !important;
                margin: 0 !important;
                border: none !important;
                padding: 0.5rem 1rem !important;
            }
            </style>
            """, unsafe_allow_html=True)

            logo_url = "https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg"
            st.image(logo_url, width=250)

            st.sidebar.header("🔍 Ask About GRANTS Analytics")
            st.sidebar.info(f"📂 Current Model: **GRANTS**")
            st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)

            if st.button("New Conversation", key="new_conversation_button"):
                start_new_conversation()

            st.markdown("### About")
            st.write(
                "This application uses **Snowflake Cortex Analyst** and **Search** to interpret "
                "your natural language questions and generate data insights. "
                "Simply ask a question below to see relevant answers and visualizations."
            )
            st.markdown("### Help & Documentation")
            st.write(
                "- [User Guide](https://docs.snowflake.com/en/guides-overview-ai-features)  \n"
                "- [Snowflake Cortex Analyst Docs](https://docs.snowflake.com/)  \n"
                "- [Contact Support](https://www.snowflake.com/en/support/)"
            )
            st.markdown("### 💡 Suggested Questions")
            for q in suggested_questions:
                if st.button(q, key=f"sidebar_suggested_{hash(q)}"):
                    st.session_state.current_query_to_process = q
                    st.session_state.selected_history_query = None
                    st.rerun()

        # Display chat history in main area
        st.title("Cortex AI Assistant for Grants")
        st.markdown(f"Semantic Model: `{SEMANTIC_MODEL.split('/')[-1]}`")

        # Display the main chat history
        unique_parent_queries = []
        for message in st.session_state.messages:
            if message["role"] == "user" and "parent_query" not in message and message["content"] not in unique_parent_queries:
                unique_parent_queries.append(message["content"])
        
        for parent_query in unique_parent_queries:
            # Display original user query
            with st.chat_message("user"):
                st.markdown(parent_query)
            
            # Find the immediate assistant response for the parent query using a safer method
            parent_assistant_message = None
            user_query_idx = -1
            
            # Find the index of the parent user query
            for idx, msg in enumerate(st.session_state.messages):
                if msg.get("role") == "user" and msg.get("content") == parent_query and "parent_query" not in msg:
                    user_query_idx = idx
                    break

            # If the user query was found and there's a message immediately after it
            if user_query_idx != -1 and (user_query_idx + 1) < len(st.session_state.messages):
                potential_assistant_response = st.session_state.messages[user_query_idx + 1]
                if potential_assistant_response.get("role") == "assistant" and "parent_query" not in potential_assistant_response:
                    parent_assistant_message = potential_assistant_response

            if parent_assistant_message:
                with st.chat_message("assistant"):
                    st.markdown(parent_assistant_message["content"])
                    if parent_assistant_message.get("sql"):
                        with st.expander("View SQL Query", expanded=False):
                            st.code(parent_assistant_message["sql"], language="sql")
                    
                    # Check if results are stored in query_results for this parent_query
                    if parent_query in st.session_state.query_results and isinstance(st.session_state.query_results[parent_query], pd.DataFrame):
                        results_df = st.session_state.query_results[parent_query]
                        st.dataframe(results_df)
                        if not results_df.empty and len(results_df.columns) >= 2:
                            st.markdown("**📈 Visualization:**")
                            with st.container(key=f"chart_container_history_{hash(parent_query)}"):
                                display_chart_tab(results_df, prefix=f"chart_history_{hash(parent_query)}", query=parent_query)

            # Display follow-up questions and their responses
            followups = [m for m in st.session_state.messages if m.get("parent_query") == parent_query]
            if followups:
                with st.expander(f"Show {len(followups)} follow-up questions for '{parent_query[:50]}...'", expanded=False):
                    for followup in followups:
                        if followup["role"] == "user":
                            with st.chat_message("user"):
                                st.markdown(f"Follow-up: {followup['content']}")
                        elif followup["role"] == "assistant":
                            with st.chat_message("assistant"):
                                st.markdown(followup["content"])

        # Handle user input or suggested question clicks
        placeholder_text = "Ask a question..."
        if st.session_state.selected_history_query:
            placeholder_text = f"Ask any follow-up question for: {st.session_state.selected_history_query}"
        
        chat_input_query = st.chat_input(placeholder=placeholder_text, key="chat_input")
        
        if chat_input_query:
            if st.session_state.selected_history_query:
                # Process as a follow-up
                process_query_and_display(chat_input_query, is_followup=True, parent_query=st.session_state.selected_history_query)
                st.session_state.selected_history_query = None # Clear selected history context
            else:
                # Process as a new query
                st.session_state.current_query_to_process = chat_input_query
                st.session_state.selected_history_query = None # Ensure no lingering history context
            st.rerun() # Rerun to update chat display immediately

        # Process selected history query or new query that was set
        if st.session_state.current_query_to_process:
            query_to_process = st.session_state.current_query_to_process
            st.session_state.current_query_to_process = None # Clear after picking it up
            if st.session_state.selected_history_query:
                st.info(f"Re-running query from history: {query_to_process}")
            process_query_and_display(query_to_process)
            st.rerun()

        # Display suggested questions in the chat area if the query fails
        if st.session_state.show_suggested_buttons:
            st.markdown("---")
            st.markdown("### 💡 Try one of these questions:")
            cols = st.columns(2)
            for idx, q in enumerate(suggested_questions):
                with cols[idx % 2]:
                    if st.button(q, key=f"chat_suggested_button_{hash(q)}"):
                        st.session_state.current_query_to_process = q
                        st.session_state.selected_history_query = None
                        st.rerun()
        
        # Display chat history in sidebar if toggled (Code 2 logic)
        if st.session_state.show_history:
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 📜 Full Chat History")
                if not st.session_state.messages:
                    st.markdown("No chat history yet.")
                else:
                    # Iterate through messages to display in sidebar history
                    # This section needs to be careful about which messages are "main" and which are follow-ups
                    displayed_main_queries = set()
                    for i, msg in enumerate(st.session_state.messages):
                        if msg["role"] == "user" and "parent_query" not in msg:
                            # Only display main user queries once
                            if msg["content"] not in displayed_main_queries:
                                user_message = msg["content"]
                                displayed_main_queries.add(user_message)

                                col_hist1, col_hist2, col_hist3 = st.columns([8, 1, 1])
                                with col_hist1:
                                    # Find the direct assistant response for this main user query
                                    assistant_response_for_history = None
                                    if (i + 1) < len(st.session_state.messages) and \
                                       st.session_state.messages[i+1]["role"] == "assistant" and \
                                       "parent_query" not in st.session_state.messages[i+1]:
                                        assistant_response_for_history = st.session_state.messages[i+1]["content"]
                                    
                                    with st.expander(f"You: {user_message}", expanded=False):
                                        if assistant_response_for_history:
                                            st.markdown(assistant_response_for_history)
                                        else:
                                            st.markdown("No direct response logged.")
                                with col_hist2:
                                    if st.button("🔍", key=f"hist_search_{i}", help="Re-run this query"):
                                        st.session_state.selected_history_query = user_message
                                        st.session_state.current_query_to_process = user_message
                                        st.rerun()
                                with col_hist3:
                                    if st.button("⬇", key=f"hist_quit_{i}", help="Set as current context"):
                                        st.session_state.selected_history_query = user_message
                                        st.session_state.show_history = False # Close history sidebar when selected
                                        st.rerun()
                        elif msg["role"] == "user" and "parent_query" in msg:
                            # Display follow-up user queries indented
                            st.markdown(f"  ↪️ Follow-up: {msg['content']}")
                        elif msg["role"] == "assistant" and "parent_query" in msg:
                            # Display follow-up assistant responses indented
                            st.markdown(f"  🤖 Follow-up Response: {msg['content']}")

    if __name__ == "__main__":
        main()
