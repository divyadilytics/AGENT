import streamlit as st
import json
import re
import requests
import snowflake.connector
import pandas as pd
from snowflake.snowpark import Session
from typing import Any, Dict, List, Optional
import plotly.express as px
from collections import Counter

# Snowflake/Cortex Configuration
HOST = "bnkzyio-ljb86662.snowflakecomputing.com"
DATABASE = "AI"
SCHEMA = "DWH_MART"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000  # in milliseconds
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.PBCS_SEARCH_SERVICE"
SEMANTIC_MODEL = '@"AI"."DWH_MART"."PBCS"/pbcs.yaml'

# Streamlit Page Config
st.set_page_config(
    page_title="PBCS Cortex AI Assistant",
    layout="wide",
    initial_sidebar_state="auto"
)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = ""
    st.session_state.password = ""
    st.session_state.CONN = None
    st.session_state.snowpark_session = None
    st.session_state.chat_history = []
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "show_suggested_buttons" not in st.session_state:
    st.session_state.show_suggested_buttons = False
if "current_query" not in st.session_state:
    st.session_state.current_query = None
if "context_cache" not in st.session_state:
    st.session_state.context_cache = {"last_query": "", "last_results": None, "last_summary": "", "last_columns": []}

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

# Suggested questions
sample_questions = [
    "Show the top 5 programs with the highest net increase in budget between FY16-17 and FY17-18.",
    "What is the total funding breakdown by account type across all organizations in FY17-18?",
    "Describe the line item summary.",
    "Describe the position summary.",
    "What is the variance between planned and actual costs by account or award?",
    "Which accounts have the highest spending within a specific department or organization?",
    "Explain the introduction of Planning and Budgeting.",
    "Show the top 10 total amounts by organization and version (only COUNCIL1 and COUNCIL2).",
    "What is the year-over-year change in total budgeted amount for each fund between FY16-17 and FY17-18?"
]

# Function to start a new conversation
def start_new_conversation():
    st.session_state.chat_history = []
    st.session_state.show_suggested_buttons = False
    st.session_state.current_query = None
    st.session_state.context_cache = {"last_query": "", "last_results": None, "last_summary": "", "last_columns": []}

# Authentication logic
if not st.session_state.authenticated:
    st.title("Snowflake Cortex AI Assistant")
    st.markdown("Please login to interact with your data")
    st.session_state.username = st.text_input("Enter Snowflake Username:", value=st.session_state.username)
    st.session_state.password = st.text_input("Enter Password:", type="password")
    if st.button("Login"):
        try:
            conn = snowflake.connector.connect(
                user=st.session_state.username,
                password=st.session_state.password,
                account="bnkzyio-ljb86662",
                host=HOST,
                port=443,
                warehouse="COMPUTE_WH",
                role="ACCOUNTADMIN",
                database=DATABASE,
                schema=SCHEMA,
            )
            st.session_state.CONN = conn
            snowpark_session = Session.builder.configs({
                "connection": conn
            }).create()
            st.session_state.snowpark_session = snowpark_session
            with conn.cursor() as cur:
                cur.execute(f"USE DATABASE {DATABASE}")
                cur.execute(f"USE SCHEMA {SCHEMA}")
                cur.execute("ALTER SESSION SET TIMEZONE = 'UTC'")
                cur.execute("ALTER SESSION SET QUOTED_IDENTIFIERS_IGNORE_CASE = TRUE")
            st.session_state.authenticated = True
            st.success("Authentication successful!")
        except Exception as e:
            st.error(f"Authentication failed: {e}")
else:
    session = st.session_state.snowpark_session

    # Utility Functions
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
            "describe", "introduction", "summary", "tell me about", "overview", "explain", "report",
            "policy", "document", "description", "highlight", "guidelines", "procedure", "how to",
            "define", "definition"
        ]
        query_lower = query.lower()
        if any(keyword in query_lower for keyword in unstructured_keywords):
            structured_score = sum(1 for keyword in structured_keywords if keyword in query_lower)
            if structured_score < 2:
                return False
        return any(keyword in query_lower for keyword in structured_keywords)

    def is_unstructured_query(query: str):
        unstructured_keywords = [
            "explain", "describe", "report", "policy", "document", "description", "summary",
            "highlight", "guidelines", "procedure", "how to", "define", "definition", "rules",
            "steps", "overview", "objective", "purpose", "benefits", "importance", "impact",
            "details", "regulation", "requirement", "compliance", "when to", "where to",
            "meaning", "interpretation", "clarify", "note", "explanation", "instructions"
        ]
        query_lower = query.lower()
        return any(word in query_lower for word in unstructured_keywords)

    def is_complete_query(query: str):
        complete_patterns = [r'\b(generate|write|create|describe|explain)\b']
        return any(re.search(pattern, query.lower()) for pattern in complete_patterns)

    def is_summarize_query(query: str):
        summarize_patterns = [r'\b(summarize|summary|condense|brief|above|previous)\b']
        return any(re.search(pattern, query.lower()) for pattern in summarize_patterns)

    def is_question_suggestion_query(query: str):
        suggestion_patterns = [
            r'\b(what|which|how)\b.*\b(questions|queries)\b.*\b(ask|can i ask)\b',
            r'\b(give me|show me|list)\b.*\b(questions|examples)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in suggestion_patterns)

    def is_follow_up_query(query: str):
        follow_up_patterns = [
            r'\b(above|previous|last|prior|that|more|tell me more|what else|further|details|only|for|from above)\b.*\b(question|query|answer|result|summary|information|about|data)\b',
            r'\b(tell me more|what else|more about|further details)\b'
        ]
        return any(re.search(pattern, query.lower()) for pattern in follow_up_patterns)

    def preprocess_query(query: str):
        query_lower = query.lower()
        tokens = query_lower.split()
        stopwords = set(['what', 'is', 'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to'])
        key_terms = [token for token in tokens if token not in stopwords and token.isalnum()]
        def normalize_term(term):
            return re.sub(r'(ing|s|ed)$', '', term)
        return [normalize_term(term) for term in key_terms]

    def summarize_unstructured_answer(answer: str, query: str):
        answer = re.sub(r"^.*?Program\sOverview", "Program Overview", answer, flags=re.DOTALL)
        sentences = re.split(r'(?<=\.|\?|\!)\s+', answer)
        sentences = [sent.strip() for sent in sentences if sent.strip()]
        if not sentences:
            return None
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
        top_sentences = [sent for sent, score in scored_sentences[:5]]
        if not top_sentences:
            top_sentences = sentences[:5]
        summary = "\n\n".join(f"• {sent}" for sent in top_sentences)
        if len(summary) < 50:  # Fallback for low-quality results
            prompt = f"Provide a concise description for the query '{query}' based on Planning and Budgeting data."
            summary = complete(prompt) or summary
        return summary

    def complete(prompt, model="mistral-large"):
        try:
            prompt = prompt.replace("'", "\\'")
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model}', '{prompt}') AS response"
            result = session.sql(query).collect()
            return result[0]["RESPONSE"]
        except Exception as e:
            if st.session_state.debug_mode:
                st.write(f"Debug: COMPLETE Error: {str(e)}")
            return None

    def summarize(text: str, query: str):
        try:
            text = re.sub(r'\s+', ' ', text.strip())
            text = text.replace("'", "\\'")
            query_sql = f"SELECT SNOWFLAKE.CORTEX.SUMMARIZE('{text}') AS summary"
            result = session.sql(query_sql).collect()
            summary = result[0]["SUMMARY"]
            if summary and len(summary) > 50:
                return summary
            raise Exception("Cortex SUMMARIZE returned empty.")
        except Exception as e:
            if st.session_state.debug_mode:
                st.write(f"Debug: SUMMARIZE Error: {str(e)}. Using fallback.")
            sentences = re.split(r'(?<=\.|\?|\!)\s+', text)
            sentences = [sent.strip() for sent in sentences if sent.strip() and len(sent) > 20]
            if not sentences:
                return None
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
            top_sentences = [sent for sent, score in scored_sentences[:3]]
            if not top_sentences:
                top_sentences = sentences[:3]
            return "\n".join(top_sentences) if top_sentences else None

    def get_historical_context(query: str):
        if not is_follow_up_query(query) or not st.session_state.chat_history:
            return None, None, None
        for message in reversed(st.session_state.chat_history):
            if message["role"] == "assistant":
                prior_query = message.get("query", "")
                content = message.get("content", "")
                results = message.get("results")
                if results is not None:
                    return prior_query, content, results
                if content and "🔍 Document Highlights" in content:
                    return prior_query, content, None
        return None, None, None

    def process_follow_up_query(query: str, prior_results: pd.DataFrame, prior_query: str):
        if prior_results is None or prior_results.empty:
            return None, "No prior data available to filter."
        
        query_lower = query.lower()
        # Tokenize query for flexible matching
        tokens = preprocess_query(query)
        
        # Identify potential entities (e.g., PG_6027, FD_1010)
        entity_pattern = r'\b(PG_\d+|FD_\d+)\b'
        entities = re.findall(entity_pattern, query_lower)
        if not entities:
            # Try token-based entity detection
            entities = [token.upper() for token in tokens if re.match(r'^(PG_|FD_)\d+$', token.upper())]
        
        # Identify target column (e.g., NET_INCREASE, AMOUNT)
        column_keywords = ['net increase', 'net_increase', 'value', 'amount', 'budget', 'change']
        target_column = None
        for keyword in column_keywords:
            if keyword in query_lower:
                target_column = 'NET_INCREASE' if 'net increase' in keyword else keyword.upper()
                break
        
        # Normalize DataFrame columns
        prior_results.columns = [col.upper() for col in prior_results.columns]
        available_columns = prior_results.columns.tolist()
        
        # Detect ID column (PROGRAM, FUND, etc.)
        id_columns = [col for col in available_columns if col in ['PROGRAM', 'FUND', 'PROGRAM_ID', 'FUND_ID']]
        if not id_columns:
            return None, "No identifiable PROGRAM or FUND column in prior results."
        id_column = id_columns[0]
        
        # If no target column specified, default to NET_INCREASE or first numeric column
        if not target_column:
            if 'NET_INCREASE' in available_columns:
                target_column = 'NET_INCREASE'
            else:
                numeric_cols = [col for col in available_columns if pd.api.types.is_numeric_dtype(prior_results[col])]
                target_column = numeric_cols[0] if numeric_cols else None
        
        if not target_column or target_column not in available_columns:
            return None, f"Column '{target_column}' not found in prior results."
        
        # Filter results
        if entities:
            filtered_results = prior_results[prior_results[id_column].str.upper().isin(entities)]
        else:
            # Fallback: filter based on tokens matching any column
            filtered_results = prior_results
            for token in tokens:
                if token.upper() in available_columns:
                    continue
                filtered_results = filtered_results[filtered_results[id_column].str.contains(token, case=False, na=False)]
        
        if filtered_results.empty:
            # Fallback SQL query if no results found
            if entities and 'NET_INCREASE' in target_column:
                entity_list = ', '.join(f"'{e}'" for e in entities)
                sql = f"""
                SELECT {id_column}, NET_INCREASE
                FROM AI.DWH_MART.PBCS
                WHERE {id_column} IN ({entity_list})
                AND FISCAL_YEAR IN ('FY16-17', 'FY17-18')
                """
                fallback_results, error = run_snowflake_query(sql)
                if fallback_results is not None and not fallback_results.empty:
                    filtered_results = fallback_results
                    if st.session_state.debug_mode:
                        st.write(f"Debug: Fallback SQL executed: {sql}")
                else:
                    return None, f"No data found for {', '.join(entities)} in prior results or database."
        
        # Select display columns
        display_columns = [id_column, target_column]
        if 'FUND_DESCRIPTION' in available_columns and id_column == 'FUND':
            display_columns.append('FUND_DESCRIPTION')
        
        result_df = filtered_results[display_columns]
        
        if st.session_state.debug_mode:
            st.write(f"Debug: Follow-Up Query - Entities: {entities}, Target Column: {target_column}, ID Column: {id_column}, Filtered Rows: {len(result_df)}")
        
        return result_df, None

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
                        if st.session_state.debug_mode:
                            st.write(f"Debug: SSE Parse Error: {str(e)}")
        return events

    def snowflake_api_call(query: str, is_structured: bool = False):
        payload = {
            "model": "mistral-large",
            "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            "tools": []
        }
        if is_structured:
            payload["tools"].append({"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}})
            payload["tool_resources"] = {"analyst1": {"semantic_model_file": SEMANTIC_MODEL}}
        else:
            payload["tools"].append({"tool_spec": {"type": "cortex_search", "name": "search1"}})
            payload["tool_resources"] = {"search1": {"name": CORTEX_SEARCH_SERVICES, "max_results": 5}}
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
            if st.session_state.debug_mode:
                st.write(f"Debug: API Response Status: {resp.status_code}")
                st.write(f"Debug: API Raw Response: {resp.text}")
            if resp.status_code < 400:
                if not resp.text.strip():
                    return None
                return parse_sse_response(resp.text)
            else:
                raise Exception(f"Failed request with status {resp.status_code}: {resp.text}")
        except Exception as e:
            if st.session_state.debug_mode:
                st.write(f"Debug: API Request Failed: {str(e)}")
            return None

    def process_sse_response(response, is_structured, query):
        sql = ""
        search_results = []
        error = None
        if not response:
            return sql, search_results, "No response from API."
        try:
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
                                            key_terms = preprocess_query(query)
                                            ranked_results = []
                                            for sr in result_data["searchResults"]:
                                                text = sr["text"]
                                                text_lower = text.lower()
                                                score = sum(1 for term in key_terms if term in text_lower)
                                                ranked_results.append((text, score))
                                            ranked_results.sort(key=lambda x: x[1], reverse=True)
                                            search_results = [
                                                summarize_unstructured_answer(text, query)
                                                for text, _ in ranked_results
                                            ]
                                            search_results = [sr for sr in search_results if sr]
            if not is_structured and not search_results:
                error = "No relevant search results returned."
        except Exception as e:
            error = f"Error Processing Response: {str(e)}"
        if st.session_state.debug_mode:
            st.write(f"Debug: Processed Response - SQL: {sql}, Search Results: {search_results}, Error: {error}")
        return sql.strip(), search_results, error

    # Visualization Function
    def display_chart_tab(df: pd.DataFrame, prefix: str, query: str):
        if df.empty or len(df.columns) < 2:
            return
        query_lower = query.lower()
        default_chart = "Bar Chart"
        if re.search(r'\b(projects)\b', query_lower):
            default_chart = "Pie Chart"
        elif re.search(r'\b(month|year|date)\b', query_lower):
            default_chart = "Line Chart"
        all_cols = list(df.columns)
        col1, col2, col3 = st.columns(3)
        x_col = col1.selectbox("X axis", all_cols, index=0, key=f"{prefix}_x")
        remaining_cols = [c for c in all_cols if c != x_col]
        y_col = col2.selectbox("Y axis", remaining_cols, index=0, key=f"{prefix}_y")
        chart_options = ["Line Chart", "Bar Chart", "Pie Chart", "Scatter Chart"]
        chart_type = col3.selectbox("Chart Type", chart_options, index=chart_options.index(default_chart), key=f"{prefix}_type")
        if chart_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col)
            st.plotly_chart(fig, key=f"{prefix}_line")
        elif chart_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col)
            st.plotly_chart(fig, key=f"{prefix}_bar")
        elif chart_type == "Pie Chart":
            fig = px.pie(df, names=x_col, values=y_col)
            st.plotly_chart(fig, key=f"{prefix}_pie")
        elif chart_type == "Scatter Chart":
            fig = px.scatter(df, x=x_col, y=y_col)
            st.plotly_chart(fig, key=f"{prefix}_scatter")

    # UI Logic
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
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=st.session_state.debug_mode)
        if st.button("New Conversation"):
            start_new_conversation()
        st.markdown("### About")
        st.write("This app uses Snowflake Cortex to answer queries about Planning and Budgeting data.")
        st.markdown("### Help")
        st.write("- [User Guide](https://docs.snowflake.com/en/guides-overview-ai-features)\n- [Contact Support](https://www.snowflake.com/en/support/)")
        st.subheader("Sample Questions")
        for sample in sample_questions:
            if st.sidebar.button(sample, key=f"sidebar_{hash(sample)}"):
                st.session_state.current_query = sample
                st.session_state.show_suggested_buttons = False

    st.title("Cortex AI Assistant by DiLytics")
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
                    display_chart_tab(message["results"], prefix=f"chart_{hash(message['content'])}", query=message.get("query", ""))

    # Chat input with fallback
    query = st.chat_input("Ask your question...", key="chat_input")
    if not query:
        st.markdown("### Fallback Input (if chat input is missing):")
        fallback_query = st.text_input("Type your question here:", key="fallback_input")
        if st.button("Submit", key="fallback_submit"):
            query = fallback_query
    if query:
        st.session_state.current_query = query
        st.session_state.show_suggested_buttons = False

    if st.session_state.current_query:
        query = st.session_state.current_query
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Generating Response..."):
                if st.session_state.debug_mode:
                    st.write(f"Debug: Query Classification - Structured: {is_structured_query(query)}, Unstructured: {is_unstructured_query(query)}, Follow-up: {is_follow_up_query(query)}")

                is_structured = is_structured_query(query)
                is_unstructured = is_unstructured_query(query)
                is_complete = is_complete_query(query)
                is_summarize = is_summarize_query(query)
                is_suggestion = is_question_suggestion_query(query)
                is_follow_up = is_follow_up_query(query)

                assistant_response = {"role": "assistant", "content": "", "query": query}
                if is_suggestion:
                    response_content = "**Suggested Questions:**\n"
                    for i, q in enumerate(sample_questions, 1):
                        response_content += f"{i}. {q}\n"
                    st.markdown(response_content)
                    assistant_response["content"] = response_content

                elif is_follow_up:
                    prior_query, prior_content, prior_results = get_historical_context(query)
                    if prior_query:
                        if st.session_state.debug_mode:
                            st.write(f"Debug: Historical Context - Prior Query: {prior_query}")
                        if is_summarize:
                            summary = summarize(prior_content or prior_query, prior_query)
                            if summary:
                                response_content = f"**Summary of '{prior_query}':**\n{summary}"
                                st.markdown(response_content)
                                assistant_response["content"] = response_content
                            else:
                                response_content = f"Could not summarize '{prior_query}'."
                                st.warning(response_content)
                                st.session_state.show_suggested_buttons = True
                                assistant_response["content"] = response_content
                        else:
                            # Process follow-up query for specific data
                            filtered_results, error = process_follow_up_query(query, prior_results, prior_query)
                            if filtered_results is not None and not filtered_results.empty:
                                response_content = f"**Results for '{query}':**"
                                st.markdown(response_content)
                                st.dataframe(filtered_results)
                                assistant_response["content"] = response_content
                                assistant_response["results"] = filtered_results
                                st.session_state.context_cache = {
                                    "last_query": query,
                                    "last_results": filtered_results,
                                    "last_summary": response_content,
                                    "last_columns": filtered_results.columns.tolist()
                                }
                            elif error:
                                if st.session_state.debug_mode:
                                    st.write(f"Debug: Follow-Up Error: {error}")
                                # Fallback to Cortex COMPLETE for vague follow-ups
                                prompt = f"Based on the previous query '{prior_query}' and its response '{prior_content[:500]}', provide more details for '{query}'."
                                response = complete(prompt)
                                if response:
                                    response_content = f"**More Information:**\n{response}"
                                    st.markdown(response_content)
                                    assistant_response["content"] = response_content
                                else:
                                    response_content = f"Could not find more information for '{query}'."
                                    st.warning(response_content)
                                    st.session_state.show_suggested_buttons = True
                                    assistant_response["content"] = response_content
                    else:
                        response_content = "No previous query found to reference."
                        st.warning(response_content)
                        st.session_state.show_suggested_buttons = True
                        assistant_response["content"] = response_content

                elif is_complete or is_unstructured:
                    if is_unstructured:
                        response = snowflake_api_call(query, is_structured=False)
                        sql, search_results, error = process_sse_response(response, is_structured=False, query=query)
                        if error:
                            response_content = f"Could not find information for '{query}'."
                            st.warning(response_content)
                            st.session_state.show_suggested_buttons = True
                            assistant_response["content"] = response_content
                        elif search_results:
                            st.markdown("**🔍 Document Highlights:**")
                            combined_results = "\n\n".join(search_results)
                            summary = summarize(combined_results, query)
                            if summary:
                                response_content = summary
                                st.markdown(response_content)
                                assistant_response["content"] = response_content
                            else:
                                response_content = f"**Key Information:**\n{search_results[0]}"
                                st.markdown(response_content)
                                assistant_response["content"] = response_content
                            st.session_state.context_cache = {
                                "last_query": query,
                                "last_results": None,
                                "last_summary": response_content,
                                "last_columns": []
                            }
                        else:
                            response_content = f"I couldn't find information for: '{query}'"
                            st.warning(response_content)
                            st.session_state.show_suggested_buttons = True
                            assistant_response["content"] = response_content
                    else:
                        response = complete(query)
                        if response:
                            response_content = f"**Generated Response:**\n{response}"
                            st.markdown(response_content)
                            assistant_response["content"] = response_content
                            st.session_state.context_cache = {
                                "last_query": query,
                                "last_results": None,
                                "last_summary": response_content,
                                "last_columns": []
                            }
                        else:
                            response_content = f"Could not generate a response for '{query}'."
                            st.warning(response_content)
                            st.session_state.show_suggested_buttons = True
                            assistant_response["content"] = response_content

                elif is_summarize:
                    summary = summarize(query, query)
                    if summary:
                        response_content = f"**Summary:**\n{summary}"
                        st.markdown(response_content)
                        assistant_response["content"] = response_content
                        st.session_state.context_cache = {
                            "last_query": query,
                            "last_results": None,
                            "last_summary": response_content,
                            "last_columns": []
                        }
                    else:
                        response_content = f"Could not generate a summary for '{query}'."
                        st.warning(response_content)
                        st.session_state.show_suggested_buttons = True
                        assistant_response["content"] = response_content

                elif is_structured:
                    response = snowflake_api_call(query, is_structured=True)
                    sql, search_results, error = process_sse_response(response, is_structured=True, query=query)
                    if error:
                        response_content = f"Could not process query '{query}'."
                        st.warning(response_content)
                        st.session_state.show_suggested_buttons = True
                        assistant_response["content"] = response_content
                    elif sql:
                        results, error = run_snowflake_query(sql)
                        if error:
                            response_content = error
                            st.warning(response_content)
                            st.session_state.show_suggested_buttons = True
                            assistant_response["content"] = response_content
                        elif results is not None and not results.empty:
                            results_text = results.to_string(index=False)
                            prompt = f"Provide a concise answer to '{query}' using:\n{results_text}"
                            summary = complete(prompt)
                            if not summary:
                                summary = "Unable to generate a summary."
                            response_content = f"**Generated Response:**\n{summary}"
                            st.markdown(response_content)
                            with st.expander("View SQL Query"):
                                st.code(sql, language="sql")
                            st.markdown(f"**Query Results ({len(results)} rows):**")
                            st.dataframe(results)
                            if len(results.columns) >= 2:
                                st.markdown("**📈 Visualization:**")
                                display_chart_tab(results, prefix=f"chart_{hash(query)}", query=query)
                            assistant_response.update({
                                "content": response_content,
                                "sql": sql,
                                "results": results,
                                "summary": summary
                            })
                            st.session_state.context_cache = {
                                "last_query": query,
                                "last_results": results,
                                "last_summary": summary,
                                "last_columns": results.columns.tolist()
                            }
                        else:
                            response_content = f"No data found for '{query}'."
                            st.warning(response_content)
                            st.session_state.show_suggested_buttons = True
                            assistant_response["content"] = response_content
                    else:
                        response_content = f"No SQL generated for '{query}'."
                        st.warning(response_content)
                        st.session_state.show_suggested_buttons = True
                        assistant_response["content"] = response_content

                else:
                    response_content = f"I couldn't understand the query: '{query}'. Try rephrasing or selecting a suggested question."
                    st.warning(response_content)
                    st.session_state.show_suggested_buttons = True
                    assistant_response["content"] = response_content

                st.session_state.chat_history.append(assistant_response)
                st.session_state.current_query = None

    # Display suggested questions in chat area if query fails
    if st.session_state.show_suggested_buttons:
        st.markdown("### Try these questions:")
        cols = st.columns(2)
        for idx, q in enumerate(sample_questions):
            with cols[idx % 2]:
                if st.button(q, key=f"chat_{hash(q)}"):
                    st.session_state.current_query = q
                    st.session_state.show_suggested_buttons = False
