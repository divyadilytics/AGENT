import streamlit as st
import json
import re
import pandas as pd
from snowflake.snowpark.context import get_active_session
from collections import Counter

# Initialize Snowflake session
session = get_active_session()

# Cortex API Configuration
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000  # in milliseconds

# Single Semantic Model Configuration
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/GRANTSyaml.yaml'
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
        text = text.replace("'", "\\'")
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

def snowflake_api_call(query: str, is_structured: bool = False, selected_model=None, is_yaml=False):
    """
    Makes an API call to Snowflake Cortex, routing to text-to-SQL or search service
    based on the query type.
    """
    payload = {
        "model": "llama3.1-70b",
        "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
        "tools": []
    }
    if is_structured or is_yaml:
        payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
        payload["tool_resources"] = {"analyst1": {"semantic_model_file": selected_model}}
    else:
        payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
        payload["tool_resources"] = {"search1": {"name": CORTEX_SEARCH_SERVICES, "max_results": 10}}
    try:
        resp = _snowflake.send_snow_api_request("POST", API_ENDPOINT, {}, {}, payload, None, API_TIMEOUT)
        response = json.loads(resp["content"])
        if st.session_state.debug_mode:
            st.write(f"Debug: API Response for query '{query}': {response}")
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
                                            search_results = [
                                                summarize_unstructured_answer(text, query)
                                                for text, _ in ranked_results
                                            ]
                                            search_results = [sr for sr in search_results if sr and "No relevant content found" not in sr]
        if not is_structured and not search_results:
            error = "No relevant search results returned from the search service."
    except Exception as e:
        error = f"❌ Error Processing Response: {str(e)}"
    if st.session_state.debug_mode:
        st.write(f"Debug: Processed Response - SQL: {sql}, Explanation: {explanation}, Search Results: {search_results}, Error: {error}")
    return sql.strip(), explanation.strip(), search_results, error

def format_results_for_history(df):
    """Formats a Pandas DataFrame into a Markdown table for chat history."""
    if df is None or df.empty:
        return "No data found."
    # Check if 'tabulate' is available, otherwise fall back to string conversion
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
        response_data = None
        
        with st.spinner("Thinking... 🤖"):
            is_structured = is_structured_query(query)
            is_yaml = detect_yaml_or_sql_intent(query)
            if is_structured or is_yaml:
                response_json, api_error = snowflake_api_call(query, is_structured=True, selected_model=SEMANTIC_MODEL, is_yaml=is_yaml)
                if api_error:
                    response_content_for_history = api_error
                    st.session_state.show_suggested_buttons = True
                else:
                    final_sql, explanation, _, sse_error = process_sse_response(response_json, is_structured=True, query=query)
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
                            response_data = results_df
                        else:
                            response_content_for_history += "⚠️ No data found for the generated SQL query.\n"
                            st.session_state.show_suggested_buttons = True
                    else:
                        response_content_for_history = "⚠️ No SQL generated. Could not understand the structured/YAML query.\n"
                        st.session_state.show_suggested_buttons = True
            else:
                response_json, api_error = snowflake_api_call(query, is_structured=False)
                if api_error:
                    response_content_for_history = api_error
                    st.session_state.show_suggested_buttons = True
                else:
                    _, _, search_results, sse_error = process_sse_response(response_json, is_structured=False, query=query)
                    if sse_error:
                        response_content_for_history = sse_error
                        st.session_state.show_suggested_buttons = True
                    elif search_results:
                        combined_results = "\n\n".join(search_results)
                        summarized_result = summarize(combined_results, query)
                        response_content_for_history += f"**🔍 Document Highlights:**\n{summarized_result}\n"
                        response_data = summarized_result
                    else:
                        response_content_for_history = f"### I couldn't find information for: '{query}'\nTry rephrasing your question or selecting from the suggested questions."
                        st.session_state.show_suggested_buttons = True
        
        st.session_state.messages.append({"role": "assistant", "content": response_content_for_history})
        if response_data is not None:
            st.session_state.query_results[query] = response_data

def display_chat_messages():
    """Displays chat messages from session state in the main chat area."""
    # This set will keep track of which parent queries have already been displayed
    displayed_parent_queries = set()

    for i, message in enumerate(st.session_state.messages):
        # Only display user messages that are not follow-ups
        if message["role"] == "user" and "parent_query" not in message:
            query = message["content"]
            if query not in displayed_parent_queries: # Prevent re-displaying the same parent query
                with st.chat_message("user"):
                    st.markdown(query)
                
                # Find the corresponding assistant message for this original query
                assistant_response = None
                # Iterate from the next message to find the direct assistant response
                for j in range(i + 1, len(st.session_state.messages)):
                    # Check if it's an assistant message and not a follow-up response
                    if st.session_state.messages[j]["role"] == "assistant" and \
                       st.session_state.messages[j].get("parent_query") is None:
                        # Assuming the first non-follow-up assistant message after a user message is its direct response
                        assistant_response = st.session_state.messages[j]["content"]
                        break
                
                # Check for follow-up questions related to this original query
                followup_messages = [
                    m for m in st.session_state.messages 
                    if m.get("parent_query") == query
                ]

                # Display the assistant's initial response if found
                if assistant_response:
                    with st.chat_message("assistant"):
                        st.markdown(assistant_response)

                # Display follow-up questions and their answers within an expander
                if followup_messages:
                    with st.expander("Follow-up Questions"):
                        for f_msg in followup_messages:
                            if f_msg["role"] == "user":
                                with st.chat_message("user"):
                                    st.markdown(f"Follow-up: {f_msg['content']}")
                            elif f_msg["role"] == "assistant":
                                with st.chat_message("assistant"):
                                    st.markdown(f_msg['content'])
                
                displayed_parent_queries.add(query) # Mark this parent query as displayed

        # Handle follow-up user messages and their assistant responses if they aren't grouped above
        # This part ensures that if a follow-up isn't perfectly linked, it still appears.
        # However, the primary grouping should happen through the 'parent_query' mechanism.
        elif message["role"] == "user" and "parent_query" in message and message["parent_query"] not in displayed_parent_queries:
            with st.chat_message("user"):
                st.markdown(f"Follow-up: {message['content']}")
            # Find the corresponding assistant message
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
                # Collect unique original user queries for history display
                history_entries = []
                for i in range(len(st.session_state.messages)):
                    msg = st.session_state.messages[i]
                    if msg["role"] == "user" and "parent_query" not in msg:
                        # Find the next assistant message that is not a follow-up
                        assistant_response_content = "No response yet."
                        for j in range(i + 1, len(st.session_state.messages)):
                            if st.session_state.messages[j]["role"] == "assistant" and "parent_query" not in st.session_state.messages[j]:
                                assistant_response_content = st.session_state.messages[j]["content"]
                                break
                        history_entries.append((msg["content"], assistant_response_content, i)) # Store (user_query, assistant_response, index)
                
                for user_query, assistant_response_content, idx in history_entries:
                    col1, col2, col3 = st.columns([8, 1, 1])
                    with col1:
                        with st.expander(user_query, expanded=(st.session_state.selected_history_query == user_query)):
                            st.markdown(assistant_response_content)
                    with col2:
                        if st.button("🔍", key=f"search_{idx}", help="Set as current context"):
                            st.session_state.selected_history_query = user_query
                            # No need to set current_query_to_process here, as chat_input will handle it.
                            st.rerun()
                    with col3:
                        # Only show "✖" if this query is currently selected as context
                        if st.session_state.selected_history_query == user_query:
                            if st.button("✖", key=f"quit_{idx}", help="Clear current context"): # Changed icon for clarity
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
        st.rerun() # Rerun to process the query immediately

    # Process selected history query or new query from chat_input
    if st.session_state.current_query_to_process:
        query_to_process = st.session_state.current_query_to_process
        st.session_state.current_query_to_process = None # Clear it after picking it up

        # If a query is being re-run from history (and not a follow-up)
        # we set selected_history_query to None to ensure it's treated as a new top-level query.
        if not st.session_state.selected_history_query: # Only clear if it's not a follow-up context
            st.session_state.selected_history_query = None
            
        process_query_and_display(query_to_process, is_followup=False) # Always false for initial processing
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
