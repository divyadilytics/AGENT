import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector
import os
from dotenv import load_dotenv
import re
import json
import _snowflake # Re-introducing this for Cortex API calls
from collections import Counter # For summarizing unstructured text

# Load environment variables (ensure your .env file is present if you intend to use it)
load_dotenv()

# Snowflake/Cortex Configuration - Using hardcoded values directly (for quick testing)
# For production, consider using os.getenv() and a .env file,
# or Streamlit secrets, or Snowflake session context for security.
HOST = "bnkzyio-ljb86662.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_TIMEOUT = 360000  # 6 minutes in milliseconds
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.Grants_search_services" # Placeholder for unstructured search
# IMPORTANT: This SEMANTIC_MODEL points to the YAML file on the stage.
# The YAML file ITSELF must define the structure of your actual data table/view.
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/GRANTSyaml.yaml' 

# These are for snowflake.connector directly, not necessarily for Cortex API token.
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
    try:
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=HOST.replace(".snowflakecomputing.com", ""),
            role=SNOWFLAKE_ROLE,
            warehouse=SNOWFLAKE_WAREHOUSE,
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
        st.write(f"Debug: Connection Attempt Details - User: {SNOWFLAKE_USER}, Role: {SNOWFLAKE_ROLE}, Warehouse: {SNOWFLAKE_WAREHOUSE}, Host: {HOST}")
    st.stop()

# Utility functions
def run_snowflake_query(query):
    """Executes a SQL query against Snowflake using snowflake.connector and returns results as a Pandas DataFrame."""
    try:
        if not query:
            return None, "⚠️ No SQL query generated."
        session.execute(query)
        df = pd.DataFrame(session.fetchall(), columns=[col[0] for col in session.description])
        return df, None
    except Exception as e:
        return None, f"❌ SQL Execution Error: {str(e)}"

def is_structured_query(query: str):
    """
    Determines if a query is structured based on keywords typically associated with data queries.
    Enhanced to reduce false positives for unstructured queries.
    """
    structured_keywords = [
        "total", "show", "top", "funding", "net increase", "net decrease", "group by", "order by",
        "how much", "give", "count", "avg", "max", "min", "least", "highest", "by year",
        "how many", "total amount", "version", "scenario", "forecast", "year", "savings",
        "award", "position", "budget", "allocation", "expenditure", "department", "variance",
        "breakdown", "comparison", "change"
    ]
    unstructured_keywords = [
        "describe", "introduction", "summary", "tell me about", "overview", "explain"
    ]
    query_lower = query.lower()
    # Check for unstructured indicators first
    if any(keyword in query_lower for keyword in unstructured_keywords):
        structured_score = sum(1 for keyword in structured_keywords if keyword in query_lower)
        if structured_score < 2:  # Require at least 2 structured keywords to override
            return False
    return any(keyword in query_lower for keyword in structured_keywords)

def is_unstructured_query(query: str):
    unstructured_keywords = [
        "policy", "document", "description", "summary", "highlight", "explain", "describe", "guidelines",
        "procedure", "how to", "define", "definition", "rules", "steps", "overview",
        "objective", "purpose", "benefits", "importance", "impact", "details", "regulation",
        "requirement", "compliance", "when to", "where to", "meaning", "interpretation",
        "clarify", "note", "explanation", "instructions"
    ]
    query_lower = query.lower()
    return any(word in query_lower for word in unstructured_keywords)

def detect_yaml_or_sql_intent(query: str):
    """Detects if a query is asking for information about the semantic model (YAML) or SQL structure."""
    yaml_keywords = ["yaml", "semantic model", "metric", "dimension", "table", "column", "sql for", "structure of"]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in yaml_keywords)

def preprocess_query(query: str):
    """Extracts key terms from the query to improve search relevance."""
    query_lower = query.lower()
    # Split on whitespace
    tokens = query_lower.split()
    
    # Remove stopwords and non-informative words
    stopwords = set(['what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to'])
    key_terms = [token for token in tokens if token not in stopwords and token.isalnum()]
    
    # Simple normalization: remove common suffixes
    def normalize_term(term):
        return re.sub(r'(ing|s|ed)$', '', term)
    
    return [normalize_term(term) for term in key_terms]

def summarize_unstructured_answer(answer: str, query: str):
    """Summarizes unstructured text by ranking sentences based on query relevance with weighted scoring."""
    # Clean the answer
    answer = re.sub(r"^.*?Program\sOverview", "Program Overview", answer, flags=re.DOTALL)
    
    # Split on periods, question marks, or exclamation marks followed by whitespace
    sentences = re.split(r'(?<=\.|\?|\!)\s+', answer)
    sentences = [sent.strip() for sent in sentences if sent.strip()]
    
    if not sentences:
        return "No relevant content found."
    
    # Extract key terms from the query
    key_terms = preprocess_query(query)
    
    # Approximate inverse document frequency (IDF) for term weighting
    all_words = ' '.join(sentences).lower().split()
    word_counts = Counter(all_words)
    total_words = len(all_words) + 1  # Avoid division by zero
    term_weights = {term: max(1.0, 3.0 * (1 - word_counts.get(term.lower(), 0) / total_words)) for term in key_terms}
    
    # Score sentences based on weighted key terms
    scored_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        score = sum(term_weights[term] for term in key_terms if term in sent_lower)
        scored_sentences.append((sent, score))
    
    # Sort sentences by score (descending) and select top 5
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [sent for sent, score in scored_sentences[:5] if score > 0]
    
    if not top_sentences:
        # Fallback to first 5 sentences
        top_sentences = sentences[:5]
    
    return "\n\n".join(f"• {sent}" for sent in top_sentences)

def summarize(text: str, query: str):
    """Calls Snowflake Cortex SUMMARIZE function with cleaned input text, with local fallback."""
    try:
        # Clean text by removing excessive whitespace and special characters
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace("'", "\\'")
        query_sql = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
        result = session.execute(query_sql).fetchall() # Using session from snowflake.connector
        summary = result[0][0] # Fetchall returns list of tuples
        if summary and len(summary) > 50:  # Ensure summary is meaningful
            return summary
        raise Exception("Cortex SUMMARIZE returned empty or too short summary.")
    except Exception as e:
        if st.session_state.debug_mode:
            st.write(f"Debug: SUMMARIZE Function Error: {str(e)}. Using local fallback.")
        # Fallback: Use relevance scoring to select top sentences
        sentences = re.split(r'(?<=\.|\?|\!)\s+', text)
        sentences = [sent.strip() for sent in sentences if sent.strip() and len(sent) > 20]
        if not sentences:
            return "No relevant content found."
        
        key_terms = preprocess_query(query)
        all_words = ' '.join(sentences).lower().split()
        word_counts = Counter(all_words)
        total_words = len(all_words) + 1
        term_weights = {term: max(1.0, 3.0 * (1 - word_counts.get(term.lower(), 0) / total_words)) for term in key_terms}
        
        scored_sentences = []
        for sent in sentences:
            sent_lower = sent.lower()
            score = sum(term_weights[term] for term in key_terms if term in sent_lower)
            scored_sentences.append((sent, score))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [sent for sent, score in scored_sentences[:3] if score > 0]
        if not top_sentences:
            top_sentences = sentences[:3]
        
        return "\n".join(top_sentences) if top_sentences else "No relevant content found."

# Re-introducing snowflake_api_call for Cortex Agent
API_ENDPOINT = "/api/v2/cortex/agent:run" # Define endpoint for Cortex API calls
def snowflake_api_call(query: str, is_structured: bool = False, selected_model=None):
    """
    Makes an API call to Snowflake Cortex, routing to text-to-SQL or search service
    based on the query type.
    """
    payload = {
        "model": "llama3.1-70b", # Using model from your previous successful Cortex call
        "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
        "tools": []
    }
    if is_structured: # This includes is_yaml intent implicitly as it's structured
        payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
        payload["tool_resources"] = {"analyst1": {"semantic_model_file": selected_model}}
    else:
        payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
        payload["tool_resources"] = {"search1": {"name": CORTEX_SEARCH_SERVICES, "max_results": 10}}
    try:
        # Use _snowflake.send_snow_api_request for Cortex calls within Streamlit in Snowflake
        # Ensure API_TIMEOUT is available globally or passed
        resp = _snowflake.send_snow_api_request("POST", API_ENDPOINT, {}, {}, payload, None, API_TIMEOUT)
        response = json.loads(resp["content"])
        if st.session_state.debug_mode:
            st.write(f"Debug: API Request Payload for query '{query}': {payload}")
            st.write(f"Debug: Raw API Response for query '{query}': {response}")
        return response, None
    except Exception as e:
        return None, f"❌ API Request Failed: {str(e)}"

def process_sse_response(response, is_structured, query):
    """
    Processes the SSE response from Snowflake Cortex, extracting SQL/explanation
    for structured queries or search results for unstructured queries.
    """
    sql = ""
    explanation = ""
    search_results = []
    error = None
    if not response:
        return sql, explanation, search_results, "No response from API."
    try:
        # The structure is often a list of events
        for event in response:
            if isinstance(event, dict) and event.get('event') == "message.delta":
                data = event.get('data', {})
                delta = data.get('delta', {})
                for content_item in delta.get('content', []):
                    if content_item.get('type') == "tool_results":
                        tool_results = content_item.get('tool_results', {})
                        if 'content' in tool_results:
                            for result in tool_results['content']:
                                if result.get('type') == 'json':
                                    result_data = result.get('json', {})
                                    if is_structured:
                                        if 'sql' in result_data:
                                            sql = result_data.get('sql', '')
                                        if 'explanation' in result_data:
                                            explanation = result_data.get('explanation', '')
                                    else: # Unstructured search results
                                        if 'searchResults' in result_data:
                                            key_terms = preprocess_query(query)
                                            ranked_results = []
                                            for sr in result_data['searchResults']:
                                                text = sr["text"]
                                                text_lower = text.lower()
                                                score = sum(1 for term in key_terms if term in text_lower)
                                                ranked_results.append((text, score))
                                            ranked_results.sort(key=lambda x: x[1], reverse=True)
                                            search_results = [
                                                summarize_unstructured_answer(text, query)
                                                for text, _ in ranked_results
                                            ]
                                            search_results = [sr for sr in search_results if sr and "No relevant content found" not in sr]
        if not is_structured and not search_results:
            error = "No relevant search results returned from the search service."
        elif is_structured and not sql and not explanation: # If structured but no SQL or explanation
            error = "No SQL generated by Cortex Analyst. Check semantic model definitions."
    except Exception as e:
        error = f"❌ Error Processing Response: {str(e)}"
    if st.session_state.debug_mode:
        st.write(f"Debug: Processed Response - SQL: {sql}, Explanation: {explanation}, Search Results: {search_results}, Error: {error}")
    return sql.strip(), explanation.strip(), search_results, error

def format_results_for_history(df):
    """Formats a Pandas DataFrame into a Markdown table for chat history."""
    if df is None or df.empty:
        return "No data found."
    if len(df.columns) == 1:
        # If it's a single value (e.g., SUM), just return the value.
        return str(df.iloc[0, 0])
    # For multiple columns, return a Markdown table.
    return df.to_markdown(index=False)

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
        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_line")
    elif chart_type == "Bar Chart":
        fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_bar")
    elif chart_type == "Pie Chart":
        fig = px.pie(df, names=x_col, values=y_col, title=f"{y_col} by {x_col}")
        st.plotly_chart(fig, use_container_width=True, key=f"{prefix}_pie")

# Main UI
st.title("Cortex AI Assistant for Grants")
st.markdown(f"Semantic Model: `{SEMANTIC_MODEL.split('/')[-1]}`")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "sql" in message and message["sql"]:
                with st.expander("View SQL Query"):
                    st.code(message["sql"], language="sql")
            if "results" in message and message["results"] is not None:
                st.markdown(f"**Query Results ({len(message['results'])} rows):**")
                st.dataframe(message["results"])
                if not message["results"].empty and len(message["results"].columns) >= 2:
                    st.markdown("**📈 Visualization:**")
                    # Use a unique key for each chart to avoid key conflicts on rerun
                    display_chart_tab(message["results"], prefix=f"chart_{hash(message.get('sql', message['content']))}", query=message['content'])

# Query input
query = st.chat_input("Ask your question...")
if query or st.session_state.current_query:
    query_to_process = query or st.session_state.current_query
    
    # Add user query to history
    st.session_state.chat_history.append({"role": "user", "content": query_to_process})
    with st.chat_message("user"):
        st.markdown(query_to_process)
    
    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            is_structured = is_structured_query(query_to_process)
            assistant_response = {"role": "assistant", "content": ""}
            
            if is_structured:
                # Call Cortex Analyst for structured queries
                response_json, api_error = snowflake_api_call(query_to_process, is_structured=True, selected_model=SEMANTIC_MODEL)
                
                if api_error:
                    st.error(api_error)
                    assistant_response["content"] = api_error
                else:
                    final_sql, explanation, _, sse_error = process_sse_response(response_json, is_structured=True, query=query_to_process)
                    
                    if sse_error:
                        st.error(sse_error)
                        assistant_response["content"] = sse_error
                    elif final_sql:
                        st.markdown("**📜 Generated SQL Query:**")
                        st.code(final_sql, language='sql')
                        assistant_response["sql"] = final_sql
                        
                        if explanation:
                            st.markdown("**📘 Explanation:**")
                            st.write(explanation)
                            assistant_response["content"] += f"**Explanation:** {explanation}\n"

                        results_df, query_error = run_snowflake_query(final_sql)
                        if query_error:
                            st.error(query_error)
                            assistant_response["content"] += f"\n❌ SQL Execution Error: {query_error}"
                        elif results_df is not None and not results_df.empty:
                            response_content = f"**📊 Results ({len(results_df)} rows):**"
                            st.markdown(response_content)
                            st.dataframe(results_df)
                            assistant_response["content"] += response_content
                            assistant_response["results"] = results_df
                        else:
                            response_content = "⚠️ No data found for the generated SQL query."
                            st.markdown(response_content)
                            assistant_response["content"] += response_content
                    else:
                        response_content = "⚠️ No SQL generated by Cortex Analyst. Please rephrase or check semantic model."
                        st.markdown(response_content)
                        assistant_response["content"] = response_content
            else:
                # Call Cortex Search for unstructured queries
                response_json, api_error = snowflake_api_call(query_to_process, is_structured=False, selected_model=None) # No semantic model for search
                if api_error:
                    st.error(api_error)
                    assistant_response["content"] = api_error
                else:
                    _, _, search_results, sse_error = process_sse_response(response_json, is_structured=False, query=query_to_process)
                    if sse_error:
                        st.error(sse_error)
                        assistant_response["content"] = sse_error
                    elif search_results:
                        st.markdown("**🔍 Document Highlights:**")
                        combined_results = "\n\n".join(search_results)
                        summarized_result = summarize(combined_results, query_to_process)
                        st.write(summarized_result)
                        assistant_response["content"] += f"**Document Highlights:**\n{summarized_result}\n"
                    else:
                        response_content = f"### I couldn't find information for: '{query_to_process}'\nTry rephrasing your question."
                        st.markdown(response_content)
                        assistant_response["content"] = response_content
            
            st.session_state.chat_history.append(assistant_response)
            st.session_state.current_query = None # Clear the triggered query
            st.rerun() # Rerun to ensure the chat history is fully updated and input cleared

# Sidebar setup
with st.sidebar:
    st.image("https://www.snowflake.com/wp-content/themes/snowflake/assets/img/logo-blue.svg", width=100)
    st.header("🔍 Ask About GRANTS Analytics")
    st.info(f"📂 Current Model: **GRANTS**") # Displays the name "GRANTSyaml.yaml"
    
    st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
    
    if st.button("New Conversation"):
        st.session_state.chat_history = []
        st.session_state.current_query = None
        st.session_state.current_results = None
        st.session_state.current_sql = None
        st.rerun()

    st.subheader("💡 Suggested Questions")
    sample_questions = [
        "What is the total actual award budget posted?",
        "What is the total actual award posted?",
        "What is the total amount of award encumbrances approved?",
        "What is the total task actual posted by award name?",
        "Tell me about the grants policy." # Example for unstructured search
    ]
    for q in sample_questions:
        if st.button(q, key=f"sidebar_suggested_{hash(q)}"):
            st.session_state.current_query = q # Set current_query to trigger processing
            st.rerun() # Rerun to immediately process the suggested question
