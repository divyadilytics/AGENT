import streamlit as st
import json
import re
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException
from collections import Counter
import uuid

# Initialize Snowflake session
try:
    session = get_active_session()
except Exception as e:
    st.error(f"Failed to initialize Snowflake session: {str(e)}")
    st.stop()

# Configuration
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/grantsyaml_27.yaml'
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.GRANTS_SEARCH_SERVICES"

st.set_page_config(page_title="Snowflake Cortex Assistant", layout="wide")
st.title("AI Assistant for GRANTS")

# Custom CSS to hide Streamlit branding
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Suggested questions
suggested_questions = [
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
    "What is this document about",
    "Subject areas",
    "Explain five layers in High level Architecture"
]

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'current_query_to_process' not in st.session_state:
    st.session_state.current_query_to_process = None
if 'show_suggested_buttons' not in st.session_state:
    st.session_state.show_suggested_buttons = False

def run_snowflake_query(query):
    """Executes a SQL query against Snowflake and returns results as a Pandas DataFrame."""
    try:
        if not query:
            return None, "No SQL query generated."
        df = session.sql(query).to_pandas()
        return df, None
    except SnowparkSQLException as e:
        return None, f"SQL Execution Error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected Error: {str(e)}"

def is_structured_query(query: str):
    """Determines if a query is structured based on data-related keywords."""
    structured_keywords = [
        "total", "show", "top", "funding", "group by", "order by", "how much", "give",
        "count", "avg", "max", "min", "by year", "how many", "total amount", "award",
        "budget", "allocation", "expenditure", "department", "variance", "breakdown"
    ]
    unstructured_keywords = ["describe", "introduction", "summary", "explain"]
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in unstructured_keywords):
        structured_score = sum(1 for keyword in structured_keywords if keyword in query_lower)
        if structured_score < 2:
            return False
    return any(keyword in query_lower for keyword in structured_keywords)

def detect_yaml_or_sql_intent(query: str):
    """Detects if a query is asking for information about the semantic model or SQL structure."""
    yaml_keywords = ["yaml", "semantic model", "metric", "dimension", "table", "column", "sql for"]
    return any(keyword in query.lower() for keyword in yaml_keywords)

def preprocess_query(query: str):
    """Extracts key terms from the query to improve search relevance."""
    query_lower = query.lower()
    tokens = query_lower.split()
    stopwords = set(['what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to'])
    key_terms = [token for token in tokens if token not in stopwords and token.isalnum()]
    def normalize_term(term):
        return re.sub(r'(ing|s|ed)$', '', term)
    return [normalize_term(term) for term in key_terms]

def summarize_unstructured_answer(answer: str, query: str):
    """Summarizes unstructured text by ranking sentences based on query relevance."""
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
    return "\n\n".join(f"* {sent}" for sent in top_sentences)

def summarize(text: str, query: str):
    """Calls Snowflake Cortex SUMMARIZE function with cleaned input text."""
    try:
        text = re.sub(r'\s+', ' ', text.strip()).replace("'", "\\'")
        query_sql = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
        result = session.sql(query_sql).collect()
        summary = result[0]["SUMMARY"]
        if summary and len(summary) > 50:
            return summary
        return summarize_unstructured_answer(text, query)
    except Exception as e:
        if st.session_state.debug_mode:
            st.write(f"Debug: SUMMARIZE Error: {str(e)}. Using local fallback.")
        return summarize_unstructured_answer(text, query)

# Line ~171 (for reference)
def snowflake_cortex_call(query: str, is_structured: bool = False, is_yaml: bool = False):
    """Uses Snowflake Cortex COMPLETE function to generate SQL or answer queries."""
    try:
        # Fetch semantic model content
        semantic_model_content = ""
        if is_structured or is_yaml:
            try:
                result = session.sql(f"SELECT GET_PATH(@{SEMANTIC_MODEL}, '')").collect()
                semantic_model_content = result[0][0] if result else "Schema not available."
            except Exception as e:
                semantic_model_content = f"Error fetching schema: {str(e)}"

        # Construct prompt based on query type
        if is_structured or is_yaml:
            prompt = (
                f"You are a SQL expert. Given the following Snowflake database schema:\n{semantic_model_content}\n"
                f"Generate a SQL query for the user request: '{query}'.\n"
                f"Return the response in JSON format with fields 'sql' (the SQL query) and 'explanation' (a brief explanation of the query)."
            )
        else:
            prompt = (
                f"Answer the following question concisely and accurately: '{query}'.\n"
                f"Return the response as a JSON object with a 'searchResults' array, where each result has a 'text' field containing the answer."
            )

        # Escape prompt for SQL
        prompt_escaped = prompt.replace("'", "\\'")
        query_sql = f"""
            SELECT SNOWFLAKE.CORTEX.COMPLETE(
                'llama3-70b',
                '{prompt_escaped}',
                500,
                0.7
            ) AS response
        """
        result = session.sql(query_sql).collect()
        content = result[0]["RESPONSE"]

        # Parse JSON response
        try:
            response_json = json.loads(content)
        except json.JSONDecodeError:
            if not is_structured:
                response_json = {"searchResults": [{"text": content.strip()}]}
            else:
                return None, f"Invalid JSON response: {content}"

        # Mimic SSE response structure
        parsed_response = {
            "event": "message.delta",
            "data": {
                "delta": {
                    "content": [{
                        "type": "tool_results",
                        "tool_results": response_json
                    }]
                }
            }
        }
        if st.session_state.debug_mode:
            st.write(f"Debug: Cortex Response for query '{query}': {parsed_response}")
        return [parsed_response], None

    except SnowparkSQLException as e:
        return None, f"Cortex SQL Error: {str(e)}"
    except Exception as e:
        return None, f"Cortex Request Failed: {str(e)}"

def process_sse_response(response, is_structured, query):
    """Processes the Cortex response, extracting SQL/explanation or search results."""
    sql = ""
    explanation = ""
    search_results = []
    error = None
    if not response:
        return sql, explanation, search_results, "No response from Cortex."
    try:
        for event in response:
            if isinstance(event, dict) and event.get('event') == "message.delta":
                data = event.get('data', {})
                delta = data.get('delta', {})
                for content_item in delta.get('content', []):
                    if content_item.get('type') == "tool_results":
                        tool_results = content_item.get('tool_results', {})
                        if is_structured:
                            if 'sql' in tool_results:
                                sql = tool_results.get('sql', '')
                            if 'explanation' in tool_results:
                                explanation = tool_results.get('explanation', '')
                        else:
                            if 'searchResults' in tool_results:
                                key_terms = preprocess_query(query)
                                ranked_results = []
                                for sr in tool_results['searchResults']:
                                    text = sr.get("text", "")
                                    text_lower = text.lower()
                                    score = sum(1 for term in key_terms if term in text_lower)
                                    ranked_results.append((text, score))
                                ranked_results.sort(key=lambda x: x[1], reverse=True)
                                search_results = [
                                    summarize_unstructured_answer(text, query)
                                    for text, _ in ranked_results if text
                                ]
                                search_results = [sr for sr in search_results if sr and "No relevant content found" not in sr]
        if not is_structured and not search_results:
            error = "No relevant search results returned from Cortex."
    except Exception as e:
        error = f"Error Processing Response: {str(e)}"
    if st.session_state.debug_mode:
        st.write(f"Debug: Processed Response - SQL: {sql}, Explanation: {explanation}, Search Results: {search_results}, Error: {error}")
    return sql.strip(), explanation.strip(), search_results, error

def format_results_for_history(df):
    """Formats a Pandas DataFrame into a Markdown table for chat history."""
    if df is None or df.empty:
        return "No data found."
    if len(df.columns) == 1:
        return str(df.iloc[0, 0])
    return df.to_markdown(index=False)

def process_query_and_display(query: str):
    """Processes a user query, interacts with Cortex, displays results, and updates session state."""
    st.session_state.show_suggested_buttons = False
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    response_content_for_history = ""

    with st.chat_message("assistant"):
        query_lower = query.lower().strip()
        if query_lower in ["hi", "hello"]:
            greeting_response = (
                "Hello! Welcome to the GRANTS AI Assistant! I'm here to help you explore and analyze "
                "grant-related data, answer questions about awards, budgets, and more, or provide insights "
                "from documents.\n\nHere are some questions you can try:\n"
                "* What is the posted budget for awards 41001, 41002, 41003, 41005, 41007, and 41018 by date?\n"
                "* Give me date-wise award breakdowns.\n"
                "* What is this document about?\n"
                "* List all subject areas.\n\n"
                "Feel free to ask anything, or pick one of the suggested questions to get started!"
            )
            st.markdown(greeting_response)
            response_content_for_history = greeting_response
            st.session_state.messages.append({"role": "assistant", "content": response_content_for_history})
            return

        with st.spinner("Thinking..."):
            is_structured = is_structured_query(query)
            is_yaml = detect_yaml_or_sql_intent(query)

            if is_structured or is_yaml:
                response_json, api_error = snowflake_cortex_call(query, is_structured=True, is_yaml=is_yaml)
                if api_error:
                    st.error(api_error)
                    response_content_for_history = api_error
                    st.session_state.show_suggested_buttons = True
                else:
                    final_sql, explanation, _, sse_error = process_sse_response(response_json, is_structured=True, query=query)
                    if sse_error:
                        st.error(sse_error)
                        response_content_for_history = sse_error
                        st.session_state.show_suggested_buttons = True
                    elif final_sql:
                        st.markdown("**SQL Query:**")
                        st.code(final_sql, language='sql')
                        response_content_for_history += f"**SQL Query:**\n```sql\n{final_sql}\n```\n"

                        if explanation:
                            st.markdown("**Explanation:**")
                            st.write(explanation)
                            response_content_for_history += f"**Explanation:**\n{explanation}\n"

                        results_df, query_error = run_snowflake_query(final_sql)
                        if query_error:
                            st.error(query_error)
                            response_content_for_history += query_error
                            st.session_state.show_suggested_buttons = True
                        elif results_df is not None and not results_df.empty:
                            st.markdown("**Results:**")
                            if len(results_df.columns) == 1:
                                st.write(f"**{results_df.iloc[0, 0]}**")
                            else:
                                st.dataframe(results_df)
                            response_content_for_history += "**Results:**\n" + format_results_for_history(results_df)
                        else:
                            st.markdown("No data found for the generated SQL query.")
                            response_content_for_history += "No data found for the generated SQL query.\n"
                            st.session_state.show_suggested_buttons = True
                    else:
                        st.markdown("No SQL generated. Could not understand the structured/YAML query.")
                        response_content_for_history = "No SQL generated. Could not understand the structured/YAML query.\n"
                        st.session_state.show_suggested_buttons = True
            else:
                response_json, api_error = snowflake_cortex_call(query, is_structured=False)
                if api_error:
                    st.error(api_error)
                    response_content_for_history = api_error
                    st.session_state.show_suggested_buttons = True
                else:
                    _, _, search_results, sse_error = process_sse_response(response_json, is_structured=False, query=query)
                    if sse_error:
                        st.error(sse_error)
                        response_content_for_history = sse_error
                        st.session_state.show_suggested_buttons = True
                    elif search_results:
                        st.markdown("**Document Highlights:**")
                        combined_results = "\n\n".join(search_results)
                        summarized_result = summarize(combined_results, query)
                        st.write(summarized_result)
                        response_content_for_history += f"**Document Highlights:**\n{summarized_result}\n"
                    else:
                        st.markdown(f"### I couldn't find information for: '{query}'")
                        st.markdown("Try rephrasing your question or selecting from the suggested questions below.")
                        response_content_for_history = f"### I couldn't find information for: '{query}'\nTry rephrasing your question or selecting from the suggested questions."
                        st.session_state.show_suggested_buttons = True

            st.session_state.messages.append({"role": "assistant", "content": response_content_for_history})

def main():
    """Main function to set up the Streamlit app."""
    st.sidebar.header("Ask About GRANTS Analytics")
    st.sidebar.info(f"Current Model: **GRANTS** (Powered by Snowflake Cortex)")
    st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)

    if st.sidebar.button("Refresh", key="new_conversation_button"):
        st.session_state.messages = []
        st.session_state.current_query_to_process = None
        st.session_state.show_suggested_buttons = False
        st.rerun()

    with st.sidebar:
        st.markdown("### Suggested Questions")
        for q in suggested_questions:
            if st.button(q, key=f"sidebar_suggested_{uuid.uuid4()}"):
                st.session_state.current_query_to_process = q
                st.rerun()

    if not st.session_state.messages:
        st.markdown("Welcome! I'm the Snowflake AI Assistant, powered by Cortex, ready to assist you with grant data analysis, summaries, and answers -- simply type your question to get started")

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    chat_input_query = st.chat_input("Ask a question...")
    if chat_input_query:
        st.session_state.current_query_to_process = chat_input_query

    if st.session_state.current_query_to_process:
        query_to_process = st.session_state.current_query_to_process
        st.session_state.current_query_to_process = None
        process_query_and_display(query_to_process)
        st.rerun()

    if st.session_state.show_suggested_buttons:
        st.markdown("---")
        st.markdown("### Try one of these questions:")
        cols = st.columns(2)
        for idx, q in enumerate(suggested_questions):
            with cols[idx % 2]:
                if st.button(q, key=f"chat_suggested_button_{uuid.uuid4()}"):
                    st.session_state.current_query_to_process = q
                    st.rerun()

if __name__ == "__main__":
    main()
