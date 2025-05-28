# Streamlit AI Assistant for GRANTS with Snowflake Cortex

import streamlit as st
import json
import re
import pandas as pd
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException
from collections import Counter

# Initialize Snowflake session
try:
    session = get_active_session()
except Exception as e:
    st.error(f"Failed to initialize Snowflake session: {str(e)}")
    st.stop()

# Config
SEMANTIC_MODEL = '@"AI"."DWH_MART"."GRANTS"/grantsyaml_27.yaml'
CORTEX_SEARCH_SERVICES = "AI.DWH_MART.GRANTS_SEARCH_SERVICES"

# Page settings
st.set_page_config(page_title="Snowflake Cortex Assistant", layout="wide")
st.title("AI Assistant for GRANTS")

# Custom styling
st.markdown("""
<style>
#MainMenu, header, footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Suggested prompts
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

# Session state setup
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Query routing
user_query = st.chat_input("Ask me anything about GRANTS data...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").markdown(user_query)
    st.chat_message("assistant").markdown("Thinking...")

    # Define helper to check query type
    def is_structured_query(q):
        keywords = ["budget", "amount", "count", "how much", "posted", "approved", "group by", "top"]
        return any(k in q.lower() for k in keywords)

    # Query processor
    def run_structured_query(q):
        prompt = f"You are a Snowflake SQL expert. Based on the schema from {SEMANTIC_MODEL}, generate a SQL query for: '{q}'. Return JSON with 'sql' and 'explanation'."
        cortex_sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'llama3-70b',
            '{prompt.replace("'", "\\'")}',
            500,
            0.7
        ) AS response
        """
        try:
            result = session.sql(cortex_sql).collect()
            response = json.loads(result[0]['RESPONSE'])
            sql_query = response.get("sql")
            explanation = response.get("explanation")
            return sql_query, explanation
        except Exception as e:
            return None, str(e)

    def run_query(sql):
        try:
            df = session.sql(sql).to_pandas()
            return df
        except Exception as e:
            st.error(f"Error executing SQL: {e}")
            return None

    def run_unstructured_query(q):
        prompt = f"Answer the following question: '{q}'. Return JSON with 'searchResults': [{{'text': '...'}}]"
        cortex_sql = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE(
            'llama3-70b',
            '{prompt.replace("'", "\\'")}',
            500,
            0.7
        ) AS response
        """
        try:
            result = session.sql(cortex_sql).collect()
            response = json.loads(result[0]['RESPONSE'])
            results = response.get("searchResults", [])
            return results
        except Exception as e:
            return [str(e)]

    # Choose path
    if is_structured_query(user_query):
        sql, explanation = run_structured_query(user_query)
        if sql:
            st.chat_message("assistant").markdown("**SQL Query:**")
            st.code(sql, language="sql")
            if explanation:
                st.markdown(f"**Explanation:** {explanation}")
            df = run_query(sql)
            if df is not None:
                st.dataframe(df)
        else:
            st.error(f"Could not generate SQL: {explanation}")
    else:
        results = run_unstructured_query(user_query)
        if results:
            for i, res in enumerate(results):
                st.markdown(f"**Answer {i+1}:** {res.get('text') if isinstance(res, dict) else res}")
