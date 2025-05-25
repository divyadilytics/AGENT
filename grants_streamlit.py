import streamlit as st
import json
import re
import pandas as pd
from snowflake.snowpark.context import get_active_session
from collections import Counter

# Initialize Snowflake session
# This line is CORRECT and will get the active session when run in Streamlit in Snowflake.
# It requires the app to be deployed within the Snowflake environment.
session = get_active_session()

# Cortex Agent UDFs Configuration
# These are the names of the UDFs you MUST create in your Snowflake environment.
# They act as wrappers for the actual Cortex API calls.
# Make sure these names match the UDFs you create in Snowflake.
CORTEX_ANALYST_AGENT_UDF = 'AI.DWH_MART.GRANTS_ANALYST_AGENT'
CORTEX_SEARCH_AGENT_UDF = 'AI.DWH_MART.GRANTS_SEARCH_AGENT'

# Single Semantic Model Configuration
# This points to your semantic model YAML file for the analyst agent
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/GRANTSyaml.yaml'
# This points to your Cortex Search Service for the search agent
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.TRAIL_SEARCH_SERVICES"

st.set_page_config(page_title="📄 Multi-Model Cortex Assistant", layout="wide")
st.title("📄 AI Assistant for GRANTS")

# Custom CSS to hide Streamlit branding, style the history button, and style the search/down buttons
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
    "list all Subjec areas"
]

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = [] # Stores all chat messages
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'current_query_to_process' not in st.session_state:
    st.session_state.current_query_to_process = None # Holds query to be processed next
if 'show_suggested_buttons' not in st.session_state:
    st.session_state.show_suggested_buttons = False
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'selected_history_query' not in st.session_state:
    st.session_state.selected_history_query = None # Stores the original query from history for follow-up
if 'query_results' not in st.session_state:
    st.session_state.query_results = {}  # Maps original question to its response data (DataFrame or text)

def run_snowflake_query(query):
    """Executes a SQL query against Snowflake and returns results as a Pandas DataFrame."""
    try:
        if not query:
            return None, "⚠️ No SQL query generated."
        df = session.sql(query)
        pandas_df = df.to_pandas()
        return pandas_df, None
    except Exception as e:
        return None, f"❌ SQL Execution Error: {str(e)}"

def is_structured_query(query: str):
    """
    Determines if a query is structured based on keywords typically associated with data queries.
    """
    structured_keywords = [
        "total", "show", "top", "funding", "net increase", "net decrease", "group by", "order by",
        "how much", "give", "count", "avg", "max", "min", "least", "highest", "by year",
        "how many", "total amount", "version", "scenario", "forecast", "year", "savings",
        "award", "position", "budget", "allocation", "expenditure", "department", "variance",
        "breakdown", "comparison", "change", "posted", "encumbrance", "actual"
    ]
    unstructured_keywords = [
        "describe", "introduction", "summary", "tell me about", "overview", "explain"
    ]
    query_lower = query.lower()

    # Check for unstructured keywords first to potentially override
    if any(keyword in query_lower for keyword in unstructured_keywords):
        structured_score = sum(1 for keyword in structured_keywords if keyword in query_lower)
        # If there are few structured keywords but unstructured ones are present, lean towards unstructured
        if structured_score < 2:
            return False

    return any(keyword in query_lower for keyword in structured_keywords)

def is_unstructured_query(query: str):
    unstructured_keywords = [
        "policy", "document", "description", "summary", "highlight", "explain", "describe", "guidelines",
        "procedure", "how to", "define", "definition", "rules", "steps", "overview",
        "objective", "purpose", "benefits", "importance", "impact", "details", "regulation",
        "requirement", "compliance", "when to", "where to", "meaning", "interpretation",
        "clarify", "note", "explanation", "instructions", "this document about", "subject areas"
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
    tokens = query_lower.split()
    stopwords = set(['what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to', 'me', 'give', 'tell', 'show'])
    key_terms = [token for token in tokens if token not in stopwords and token.isalnum()]
    def normalize_term(term):
        return re.sub(r'(ing|s|ed)$', '', term)
    return [normalize_term(term) for term in key_terms]

def summarize_unstructured_answer(answer: str, query: str):
    """Summarizes unstructured text by ranking sentences based on query relevance with weighted scoring."""
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

def summarize(text: str, query: str):
    """Calls Snowflake Cortex SUMMARIZE function with cleaned input text, with local fallback."""
    try:
        text = re.sub(r'\s+', ' ', text.strip())
        text = text.replace("'", "\\'") # Escape single quotes for SQL
        query_sql = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
        result = session.sql(query_sql).collect()
        summary = result[0]["SUMMARY"]
        if summary and len(summary) > 50:
            return summary
        raise Exception("Cortex SUMMARIZE returned empty or too short summary.")
    except Exception as e:
        if st.session_state.debug_mode:
            st.write(f"Debug: SUMMARIZE Function Error: {str(e)}. Using local fallback.")
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

def call_cortex_agent_udf(agent_udf_name: str, payload: dict):
    """
    Calls a Snowflake Cortex Agent UDF (User-Defined Function) with a given payload.
    The UDF is responsible for making the actual Cortex API call.
    """
    error = None
    try:
        # Convert the Python dictionary payload to a JSON string for the UDF input
        # Escape single quotes within the JSON string for SQL compatibility
        payload_json_str = json.dumps(payload).replace("'", "''")

        # Construct the SQL CALL statement for the agent UDF
        call_sql = f"""
        SELECT {agent_udf_name}(PARSE_JSON('{payload_json_str}')) AS RESULT;
        """
        if st.session_state.debug_mode:
            st.write(f"Debug: Calling UDF with SQL: {call_sql}")

        # Execute the SQL CALL statement
        # The result of a UDF call is typically a single row, single column (VARIANT or VARCHAR)
        result = session.sql(call_sql).collect()

        if result and result[0][0]:
            # The UDF is expected to return the raw JSON response from Cortex
            # as a VARIANT. It should already be a Python dict if returned as VARIANT.
            return result[0][0], None
        else:
            error = f"No valid response from Cortex Agent UDF: {agent_udf_name}"
            return None, error

    except Exception as e:
        error = f"❌ Error calling Cortex Agent UDF '{agent_udf_name}': {str(e)}"
        return None, error

def process_cortex_response(response_data, is_structured, query):
    """
    Processes the raw JSON response from the Cortex Agent UDF, extracting
    SQL/explanation for structured queries or search results for unstructured queries.
    This function expects a parsed JSON dictionary.
    """
    sql = ""
    explanation = ""
    search_results = []
    error = None

    if not response_data:
        return sql, explanation, search_results, "No valid response data to process."

    try:
        # The structure of the 'response_data' depends on the exact output of your Cortex Agent UDFs.
        # This parsing logic assumes a common structure where tool results are nested.
        # You might need to adjust this part based on what your actual UDF returns.

        if 'messages' in response_data and isinstance(response_data['messages'], list):
            for message in response_data['messages']:
                # Cortex responses can have 'tool' role for tool outputs or 'assistant' for direct answers
                if message.get('role') in ['tool', 'assistant']:
                    for content_item in message.get('content', []):
                        if content_item.get('type') == 'tool_results':
                            tool_results = content_item.get('tool_results', {})
                            if 'content' in tool_results and isinstance(tool_results['content'], list):
                                for result in tool_results['content']:
                                    if result.get('type') == 'json' and 'json' in result:
                                        data_from_tool = result['json']
                                        if is_structured:
                                            sql = data_from_tool.get('sql', '')
                                            explanation = data_from_tool.get('explanation', '')
                                        else:
                                            if 'searchResults' in data_from_tool and isinstance(data_from_tool['searchResults'], list):
                                                key_terms = preprocess_query(query)
                                                ranked_results = []
                                                for sr in data_from_tool['searchResults']:
                                                    text = sr.get('text', '')
                                                    text_lower = text.lower()
                                                    score = sum(1 for term in key_terms if term in text_lower)
                                                    ranked_results.append((text, score))
                                                ranked_results.sort(key=lambda x: x[1], reverse=True)
                                                search_results = [
                                                    summarize_unstructured_answer(text, query)
                                                    for text, _ in ranked_results
                                                ]
                                                search_results = [sr for sr in search_results if sr and "No relevant content found" not in sr]
        
        # If the main assistant message contains a direct text response (e.g., for simple questions)
        # This handles cases where Cortex might just give a direct text answer without tool results
        elif 'answer' in response_data and isinstance(response_data['answer'], str) and not is_structured and not search_results:
            search_results.append(response_data['answer'])

        if not is_structured and not sql and not explanation and not search_results:
            error = "No relevant response (SQL, explanation, or search results) returned from the Cortex agent."

    except Exception as e:
        error = f"❌ Error parsing Cortex response: {str(e)}. Raw data: {response_data}"

    if st.session_state.debug_mode:
        st.write(f"Debug: Processed Response - SQL: {sql}, Explanation: {explanation}, Search Results: {search_results}, Error: {error}")
    return sql.strip(), explanation.strip(), search_results, error

def format_results_for_history(df):
    """Formats a Pandas DataFrame into a Markdown table for chat history."""
    if df is None or df.empty:
        return "No data found."
    try:
        return df.to_markdown(index=False)
    except ImportError:
        st.warning("`tabulate` library not found. Please install it (`pip install tabulate`) for better table formatting.")
        return df.to_string(index=False) # Fallback to default string representation

def process_followup_query(followup_query: str, parent_query: str):
    """
    Processes a follow-up query using only the parent query's response data.
    """
    if parent_query not in st.session_state.query_results:
        return f"⚠️ No response data available for parent query: {parent_query}"

    response_data = st.session_state.query_results[parent_query]
    response_content = ""

    if isinstance(response_data, pd.DataFrame):
        # Structured query follow-up
        df = response_data
        followup_lower = followup_query.lower()

        # Extract key terms and potential identifiers
        key_terms = preprocess_query(followup_query)
        award_number = None
        for term in key_terms:
            if term.isdigit() or (len(term) >= 4 and any(c.isdigit() for c in term)):  # Likely an award number
                award_number = term
                break

        # Find columns based on query terms
        requested_column = None
        for term in key_terms:
            if term in ['award', 'number', 'id', 'no']:  # Skip identifier terms
                continue
            for col in df.columns:
                if term in col.lower():
                    requested_column = col
                    break
            if requested_column:
                break

        # Find award number column
        award_number_col = None
        for col in df.columns:
            if 'award' in col.lower() and ('number' in col.lower() or 'id' in col.lower() or 'no' in col.lower()):
                award_number_col = col
                break

        if requested_column:
            if award_number and award_number_col:
                # Filter by award number (case-insensitive, partial match)
                award_number_str = str(award_number).lower().replace('l', '1').replace('o', '0')
                result = df[df[award_number_col].astype(str).str.lower().str.contains(award_number_str, na=False)]
                if not result.empty:
                    value = result[requested_column].iloc[0]
                    response_content = f"**{requested_column}** for award {award_number}: {value}"
                else:
                    available_awards = df[award_number_col].astype(str).unique() if award_number_col else ['None']
                    response_content = f"No data found for award {award_number} in the parent response. Available awards: {', '.join(available_awards)}"
            else:
                # Return all values for the column
                response_content = f"**{requested_column}** values:\n{df[requested_column].to_markdown(index=False)}"
        else:
            available_columns = ', '.join(df.columns)
            response_content = f"No relevant column found for '{followup_query}' in the parent response. Available columns: {available_columns}"

    else:
        # Unstructured query follow-up
        parent_text = response_data
        response_content = summarize(parent_text, followup_query)
        if response_content == "No relevant content found.":
            response_content = f"No relevant information found for '{followup_query}' in the parent response."

    if st.session_state.debug_mode:
        st.write(f"Debug: Follow-up query '{followup_query}' for parent '{parent_query}' - Response: {response_content}")

    return response_content

def process_query_and_display(query: str, is_followup: bool = False, parent_query: str = None):
    """
    Processes a user query or follow-up, interacts with Cortex for new queries,
    and updates session state.
    """
    st.session_state.show_suggested_buttons = False

    if is_followup:
        st.session_state.messages.append({"role": "user", "content": query, "parent_query": parent_query})
        response_content = process_followup_query(query, parent_query)
        st.session_state.messages.append({"role": "assistant", "content": response_content, "parent_query": parent_query})
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        response_content_for_history = ""
        response_data_for_storage = None # This will store the DF or text for follow-ups

        with st.spinner("Thinking... 🤖"):
            is_structured = is_structured_query(query)
            is_yaml = detect_yaml_or_sql_intent(query)
            
            cortex_payload = {
                "model": "llama3.1-70b", # Or your preferred model
                "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
                "tools": []
            }

            if is_structured or is_yaml:
                cortex_payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
                cortex_payload["tool_resources"] = {"analyst1": {"semantic_model_file": SEMANTIC_MODEL}}
                
                cortex_response_json, api_error = call_cortex_agent_udf(CORTEX_ANALYST_AGENT_UDF, cortex_payload)
                if api_error:
                    response_content_for_history = api_error
                    st.session_state.show_suggested_buttons = True
                else:
                    final_sql, explanation, _, sse_error = process_cortex_response(cortex_response_json, is_structured=True, query=query)
                    if sse_error:
                        response_content_for_history = sse_error
                        st.session_state.show_suggested_buttons = True
                    elif final_sql:
                        response_content_for_history += f"**📜 SQL Query:**\n```sql\n{final_sql}\n```\n"
                        if explanation:
                            response_content_for_history += f"**📘 Explanation:**\n{explanation}\n"
                        results_df, query_error = run_snowflake_query(final_sql)
                        if query_error:
                            response_content_for_history += query_error
                            st.session_state.show_suggested_buttons = True
                        elif results_df is not None and not results_df.empty:
                            response_content_for_history += "**📊 Results:**\n" + format_results_for_history(results_df)
                            response_data_for_storage = results_df # Store DataFrame for follow-up
                        else:
                            response_content_for_history += "⚠️ No data found for the generated SQL query.\n"
                            st.session_state.show_suggested_buttons = True
                    else:
                        response_content_for_history = "⚠️ No SQL generated. Could not understand the structured/YAML query.\n"
                        st.session_state.show_suggested_buttons = True
            else: # Unstructured query
                cortex_payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
                cortex_payload["tool_resources"] = {"search1": {"name": CORTEX_SEARCH_SERVICES, "max_results": 10}}
                
                cortex_response_json, api_error = call_cortex_agent_udf(CORTEX_SEARCH_AGENT_UDF, cortex_payload)
                if api_error:
                    response_content_for_history = api_error
                    st.session_state.show_suggested_buttons = True
                else:
                    _, _, search_results, sse_error = process_cortex_response(cortex_response_json, is_structured=False, query=query)
                    if sse_error:
                        response_content_for_history = sse_error
                        st.session_state.show_suggested_buttons = True
                    elif search_results:
                        combined_results = "\n\n".join(search_results)
                        summarized_result = summarize(combined_results, query)
                        response_content_for_history += f"**🔍 Document Highlights:**\n{summarized_result}\n"
                        response_data_for_storage = summarized_result # Store text for follow-up
                    else:
                        response_content_for_history = f"### I couldn't find information for: '{query}'\nTry rephrasing your question or selecting from the suggested questions."
                        st.session_state.show_suggested_buttons = True

        st.session_state.messages.append({"role": "assistant", "content": response_content_for_history})
        if response_data_for_storage is not None:
            st.session_state.query_results[query] = response_data_for_storage

def display_chat_messages():
    """Displays chat messages from session state in the main chat area."""
    displayed_parent_queries = set()

    for i, message in enumerate(st.session_state.messages):
        if message["role"] == "user" and "parent_query" not in message:
            query = message["content"]
            if query not in displayed_parent_queries:
                with st.chat_message("user"):
                    st.markdown(query)

                assistant_response = None
                for j in range(i + 1, len(st.session_state.messages)):
                    if st.session_state.messages[j]["role"] == "assistant" and \
                       st.session_state.messages[j].get("parent_query") is None:
                        assistant_response = st.session_state.messages[j]["content"]
                        break

                if assistant_response:
                    with st.chat_message("assistant"):
                        st.markdown(assistant_response)

                followup_messages = [
                    m for m in st.session_state.messages
                    if m.get("parent_query") == query
                ]

                if followup_messages:
                    with st.expander("Follow-up Questions"):
                        for f_msg in followup_messages:
                            if f_msg["role"] == "user":
                                with st.chat_message("user"):
                                    st.markdown(f"Follow-up: {f_msg['content']}")
                            elif f_msg["role"] == "assistant":
                                with st.chat_message("assistant"):
                                    st.markdown(f_msg['content'])

                displayed_parent_queries.add(query)

        elif message["role"] == "user" and "parent_query" in message and message["parent_query"] not in displayed_parent_queries:
            with st.chat_message("user"):
                st.markdown(f"Follow-up: {message['content']}")
            for j in range(i + 1, len(st.session_state.messages)):
                if st.session_state.messages[j]["role"] == "assistant" and \
                   st.session_state.messages[j].get("parent_query") == message["content"]:
                    with st.chat_message("assistant"):
                        st.markdown(st.session_state.messages[j]["content"])
                    break

def main():
    # History button in top right corner
    col1, col2 = st.columns([9, 1])
    with col2:
        if st.button("📜 Toggle History", key="history_toggle", type="primary"):
            st.session_state.show_history = not st.session_state.show_history
            st.session_state.selected_history_query = None  # Reset selected query
            st.rerun() # Rerun to update sidebar visibility

    # Sidebar setup
    st.sidebar.header("🔍 Ask About GRANTS Analytics")
    st.sidebar.info(f"📂 Current Model: **GRANTS**")
    st.session_state.debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)

    if st.sidebar.button("Refresh", key="new_conversation_button"):
        st.session_state.messages = []
        st.session_state.current_query_to_process = None
        st.session_state.show_suggested_buttons = False
        st.session_state.show_history = False
        st.session_state.selected_history_query = None
        st.session_state.query_results = {}
        st.rerun()

    # Display chat history in sidebar if toggled
    if st.session_state.show_history:
        with st.sidebar:
            st.markdown("### 📜 Chat History")
            if not st.session_state.messages:
                st.markdown("No chat history yet.")
            else:
                history_entries = []
                for i in range(len(st.session_state.messages)):
                    msg = st.session_state.messages[i]
                    if msg["role"] == "user" and "parent_query" not in msg:
                        assistant_response_content = "No response yet."
                        for j in range(i + 1, len(st.session_state.messages)):
                            if st.session_state.messages[j]["role"] == "assistant" and "parent_query" not in st.session_state.messages[j]:
                                assistant_response_content = st.session_state.messages[j]["content"]
                                break
                        history_entries.append((msg["content"], assistant_response_content, i))

                for user_query, assistant_response_content, idx in history_entries:
                    col1, col2, col3 = st.columns([8, 1, 1])
                    with col1:
                        with st.expander(user_query, expanded=(st.session_state.selected_history_query == user_query)):
                            st.markdown(assistant_response_content)
                    with col2:
                        if st.button("🔍", key=f"search_{idx}", help="Set as current context"):
                            st.session_state.selected_history_query = user_query
                            st.rerun()
                    with col3:
                        if st.session_state.selected_history_query == user_query:
                            if st.button("✖", key=f"quit_{idx}", help="Clear current context"):
                                st.session_state.selected_history_query = None
                                st.rerun()

    with st.sidebar:
        st.markdown("### 💡 Suggested Questions")
        for q in suggested_questions:
            if st.button(q, key=f"sidebar_suggested_{hash(q)}"):
                st.session_state.current_query_to_process = q
                st.session_state.selected_history_query = None
                st.rerun()

    # Display chat messages in the main area
    display_chat_messages()

    # Handle user input or suggested question clicks
    placeholder_text = "Ask a question..."
    if st.session_state.selected_history_query:
        placeholder_text = f"Ask any follow-up question for: '{st.session_state.selected_history_query}'"

    chat_input_query = st.chat_input(placeholder=placeholder_text, key="chat_input")

    if chat_input_query:
        if st.session_state.selected_history_query:
            process_query_and_display(chat_input_query, is_followup=True, parent_query=st.session_state.selected_history_query)
        else:
            st.session_state.current_query_to_process = chat_input_query
        st.rerun()

    # Process selected history query or new query from chat_input
    if st.session_state.current_query_to_process:
        query_to_process = st.session_state.current_query_to_process
        st.session_state.current_query_to_process = None

        if not st.session_state.selected_history_query:
            st.session_state.selected_history_query = None

        process_query_and_display(query_to_process, is_followup=False)
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

if __name__ == "__main__":
    main()
