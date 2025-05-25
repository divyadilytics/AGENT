import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector
from collections import Counter
import re
import os
from dotenv import load_dotenv

# Load environment variables locally
load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="📄 Multi-Model Cortex Assistant", layout="wide")
st.title("📄 AI Assistant for GRANTS")

# Custom CSS (unchanged)
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
.stButton button[kind="primary"] {
    position: fixed;
    top: 10px;
    right: 10px;
    z-index: 1000;
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    padding: 8px 16px;
}
.stExpander {
    border: none;
    background-color: transparent;
}
.stExpander > div > div > button::before {
    content: "▶ ";
    color: #4CAF50;
}
.stExpander[open] > div > div > button::before {
    content: "▼ ";
    color: #4CAF50;
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
    color: #4CAF50;
}
</style>
""", unsafe_allow_html=True)

# Suggested questions (unchanged)
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
    "list all Subjec areas"
]

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
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
    st.session_state.query_results = {}

# Snowflake connection
@st.cache_resource
def init_snowflake_connection():
    """Initialize Snowflake connection with error handling."""
    required_env_vars = [
        "SNOWFLAKE_ACCOUNT",
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
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            role=os.getenv("SNOWFLAKE_ROLE"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
            database="AI",
            schema="DWH_MART"
        )
        return conn
    except Exception as e:
        raise Exception(f"Failed to connect to Snowflake: {str(e)}")

# Initialize Snowflake connection
try:
    conn = init_snowflake_connection()
    session = conn.cursor()
except Exception as e:
    st.error(str(e))
    if st.session_state.debug_mode:
        st.write(f"Debug: Environment variables - ACCOUNT: {os.getenv('SNOWFLAKE_ACCOUNT')}, USER: {os.getenv('SNOWFLAKE_USER')}, ROLE: {os.getenv('SNOWFLAKE_ROLE')}, WAREHOUSE: {os.getenv('SNOWFLAKE_WAREHOUSE')}")
    st.stop()

def run_snowflake_query(query):
    """Executes a SQL query against Snowflake and returns results as a Pandas DataFrame."""
    try:
        if not query:
            return None, "⚠️ No SQL query generated."
        session.execute(query)
        df = pd.DataFrame(session.fetchall(), columns=[col[0] for col in session.description])
        return df, None
    except Exception as e:
        return None, f"❌ SQL Execution Error: {str(e)}"

def create_chart(df, x_col=None, y_col=None):
    """Creates an interactive Plotly chart from a DataFrame."""
    if df is None or df.empty:
        return None
    if x_col is None:
        x_col = df.columns[0] if len(df.columns) > 0 else None
    if y_col is None:
        y_col = df.columns[1] if len(df.columns) > 1 else None
    if x_col and y_col:
        try:
            fig = px.bar(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
            return fig
        except Exception as e:
            if st.session_state.debug_mode:
                st.write(f"Debug: Chart creation failed: {str(e)}")
            return None
    return None

def is_structured_query(query: str):
    """Determines if a query is structured based on keywords."""
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
    if any(keyword in query_lower for keyword in unstructured_keywords):
        structured_score = sum(1 for keyword in structured_keywords if keyword in query_lower)
        if structured_score < 2:
            return False
    return any(keyword in query_lower for keyword in structured_keywords)

def preprocess_query(query: str):
    """Extracts key terms from the query."""
    query_lower = query.lower()
    tokens = query_lower.split()
    stopwords = set(['what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to'])
    key_terms = [token for token in tokens if token not in stopwords and token.isalnum()]
    def normalize_term(term):
        return re.sub(r'(ing|s|ed)$', '', term)
    return [normalize_term(term) for term in key_terms]

def summarize_unstructured_answer(answer: str, query: str):
    """Summarizes unstructured text."""
    answer = re.sub(r"^.*?Program\sOverview", "Program Overview", answer, flags=re.DOTALL)
    sentences = re.split(r'(?<=\.|\?|\!)\s+', answer)
    sentences = [sent.strip() for sent in sentences if sent.strip()]
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
    top_sentences = [sent for sent, score in scored_sentences[:5] if score > 0]
    if not top_sentences:
        top_sentences = sentences[:5]
    return "\n\n".join(f"• {sent}" for sent in top_sentences)

def format_results_for_history(df):
    """Formats a DataFrame into a Markdown table."""
    if df is None or df.empty:
        return "No data found."
    if len(df.columns) == 1:
        return str(df.iloc[0, 0])
    return df.to_markdown(index=False)

def process_followup_query(followup_query: str, parent_query: str):
    """Processes a follow-up query using parent query's response data."""
    if parent_query not in st.session_state.query_results:
        return f"⚠️ No response data available for parent query: {parent_query}"
    
    response_data = st.session_state.query_results[parent_query]
    response_content = ""
    
    if isinstance(response_data, pd.DataFrame):
        df = response_data
        followup_lower = followup_query.lower()
        key_terms = preprocess_query(followup_query)
        award_number = None
        for term in key_terms:
            if term.isdigit() or (len(term) >= 4 and any(c.isdigit() for c in term)):
                award_number = term
                break
        columns = [col.lower() for col in df.columns]
        requested_column = None
        for term in key_terms:
            if term in ['award', 'number', 'id', 'no']:
                continue
            for col in df.columns:
                if term in col.lower():
                    requested_column = col
                    break
            if requested_column:
                break
        award_number_col = None
        for col in df.columns:
            if 'award' in col.lower() and ('number' in col.lower() or 'id' in col.lower() or 'no' in col.lower()):
                award_number_col = col
                break
        if requested_column:
            if award_number and award_number_col:
                award_number_str = str(award_number).lower().replace('l', '1').replace('o', '0')
                result = df[df[award_number_col].astype(str).str.lower().str.contains(award_number_str, na=False)]
                if not result.empty:
                    value = result[requested_column].iloc[0]
                    response_content = f"**{requested_column}** for award {award_number}: {value}"
                else:
                    available_awards = df[award_number_col].astype(str).unique() if award_number_col else ['None']
                    response_content = f"No data found for award {award_number}. Available awards: {', '.join(available_awards)}"
            else:
                response_content = f"**{requested_column}** values:\n{df[requested_column].to_markdown(index=False)}"
        else:
            available_columns = ', '.join(df.columns)
            response_content = f"No relevant column found for '{followup_query}'. Available columns: {available_columns}"
    else:
        response_content = summarize_unstructured_answer(response_data, followup_query)
        if response_content == "No relevant content found.":
            response_content = f"No relevant information found for '{followup_query}'."
    
    if st.session_state.debug_mode:
        st.write(f"Debug: Follow-up query '{followup_query}' for parent '{parent_query}' - Response: {response_content}")
    
    return response_content

def process_query_and_display(query: str, is_followup: bool = False, parent_query: str = None):
    """Processes a user query and displays results with a chart."""
    st.session_state.show_suggested_buttons = False
    
    if is_followup and parent_query:
        st.session_state.messages.append({"role": "user", "content": query, "parent_query": parent_query})
        with st.chat_message("user"):
            st.markdown(f"Follow-up: {query}")
        with st.chat_message("assistant"):
            response_content = process_followup_query(query, parent_query)
            st.markdown(response_content)
            st.session_state.messages.append({"role": "assistant", "content": response_content, "parent_query": parent_query})
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        response_content_for_history = ""
        response_data = None
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤖"):
                is_structured = is_structured_query(query)
                if is_structured:
                    # Simplified SQL generation (replace with your actual queries or Cortex API)
                    sample_sql = f"SELECT * FROM AI.DWH_MART.GRANTS WHERE QUERY_TEXT ILIKE '%{query}%' LIMIT 10"
                    if st.session_state.debug_mode:
                        st.write(f"Debug: Generated SQL: {sample_sql}")
                    st.markdown("**📜 SQL Query:**")
                    st.code(sample_sql, language='sql')
                    response_content_for_history += f"**📜 SQL Query:**\n```sql\n{sample_sql}\n```\n"
                    results_df, query_error = run_snowflake_query(sample_sql)
                    if query_error:
                        st.error(query_error)
                        response_content_for_history += query_error
                        st.session_state.show_suggested_buttons = True
                    elif results_df is not None and not results_df.empty:
                        st.markdown("**📊 Results:**")
                        st.dataframe(results_df)
                        response_content_for_history += "**📊 Results:**\n" + format_results_for_history(results_df)
                        chart = create_chart(results_df)
                        if chart:
                            st.markdown("**📈 Chart:**")
                            st.plotly_chart(chart, use_container_width=True)
                            response_content_for_history += "**📈 Chart:**\n[Interactive chart displayed]\n"
                        response_data = results_df
                    else:
                        st.markdown("⚠️ No data found for the query.")
                        response_content_for_history += "⚠️ No data found for the query.\n"
                        st.session_state.show_suggested_buttons = True
                else:
                    response_content = f"Summary for '{query}': This is a placeholder response for unstructured queries."
                    st.write(response_content)
                    response_content_for_history += f"**🔍 Summary:**\n{response_content}\n"
                    response_data = response_content
                    st.session_state.show_suggested_buttons = True
                st.session_state.messages.append({"role": "assistant", "content": response_content_for_history})
                if response_data is not None:
                    st.session_state.query_results[query] = response_data

def main():
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("📜 Toggle History", key="history_toggle", type="primary"):
            st.session_state.show_history = not st.session_state.show_history
            st.session_state.selected_history_query = None

    st.sidebar.header("🔍 Ask About GRANTS Analytics")
    st.sidebar.info("📂 Current Model: **GRANTS**")
    st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)

    if st.sidebar.button("Refresh", key="new_conversation_button"):
        st.session_state.clear()
        st.rerun()

    if st.session_state.show_history:
        with st.sidebar:
            st.markdown("### 📜 Chat History")
            if not st.session_state.messages:
                st.markdown("No chat history yet.")
            else:
                for i in range(0, len(st.session_state.messages), 2):
                    if i + 1 < len(st.session_state.messages) and \
                       st.session_state.messages[i]["role"] == "user" and \
                       st.session_state.messages[i + 1]["role"] == "assistant" and \
                       "parent_query" not in st.session_state.messages[i]:
                        user_message = st.session_state.messages[i]["content"]
                        assistant_message = st.session_state.messages[i + 1]["content"]
                        col1, col2, col3 = st.columns([8, 1, 1])
                        with col1:
                            with st.expander(user_message, expanded=False):
                                st.markdown(assistant_message)
                        with col2:
                            if st.button("🔍", key=f"search_{i}", help="Re-run this query"):
                                st.session_state.selected_history_query = user_message
                                st.session_state.current_query_to_process = user_message
                                st.rerun()
                        with col3:
                            if st.button("⬇", key=f"quit_{i}", help="Quit this chat context"):
                                st.session_state.selected_history_query = None
                                st.session_state.current_query_to_process = None
                                st.rerun()

    with st.sidebar:
        st.markdown("### 💡 Suggested Questions")
        for q in suggested_questions:
            if st.button(q, key=f"sidebar_suggested_{hash(q)}"):
                st.session_state.current_query_to_process = q
                st.session_state.selected_history_query = None
                st.rerun()

    parent_queries = set()
    for message in st.session_state.messages:
        if message["role"] == "user" and "parent_query" not in message:
            parent_queries.add(message["content"])
    
    for parent_query in parent_queries:
        with st.chat_message("user"):
            st.markdown(parent_query)
        assistant_messages = [m for m in st.session_state.messages if m["role"] == "assistant" and "parent_query" not in m and m["content"] == next(m2["content"] for m2 in st.session_state.messages if m2["role"] == "assistant" and m2.get("parent_query") == parent_query or (m2["content"] == m["content"] and m2.get("parent_query") is None))]
        if assistant_messages:
            with st.chat_message("assistant"):
                st.markdown(assistant_messages[0]["content"])
        followups = [m for m in st.session_state.messages if m.get("parent_query") == parent_query]
        if followups:
            with st.expander("Follow-up Questions"):
                for followup in followups:
                    if followup["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(f"Follow-up: {followup['content']}")
                    elif followup["role"] == "assistant":
                        with st.chat_message("assistant"):
                            st.markdown(followup["content"])

    placeholder_text = "Ask a question..."
    if st.session_state.selected_history_query:
        placeholder_text = f"Ask any follow-up question for: {st.session_state.selected_history_query}"
    chat_input_query = st.chat_input(placeholder=placeholder_text, key="chat_input")
    if chat_input_query:
        if st.session_state.selected_history_query:
            process_query_and_display(chat_input_query, is_followup=True, parent_query=st.session_state.selected_history_query)
            st.session_state.selected_history_query = None
        else:
            st.session_state.current_query_to_process = chat_input_query
            st.session_state.selected_history_query = None

    if st.session_state.current_query_to_process:
        query_to_process = st.session_state.current_query_to_process
        st.session_state.current_query_to_process = None
        if st.session_state.selected_history_query:
            st.info(f"Re-running query: {query_to_process}")
        process_query_and_display(query_to_process)
        st.rerun()

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

if __name__ == "__main__":
    main()
