# streamlit_app.py
import streamlit as st
from RAG_agent import *  # Import your LangGraph flow
from langchain_core.runnables import RunnableConfig
from crawler_v4_robots import *

st.set_page_config(page_title="Company Goal Analyzer", layout="centered")

st.title("🔍 Company Goal Analyzer")
st.write("Enter the company website URL below to learn about its goal.")

company_url = st.text_input("🌐 Company URL", placeholder="https://www.example.com")

if st.button("Analyze"):
    if not company_url.strip():
        st.warning("Please enter a valid company URL.")
    else:
        # Construct input state
        # crawl_my_website = main(company_url)
        input_state = {
            "company_url": company_url,
            "query": "What is the goal of the company?",
            "summary" : "The name of the company is Product Space.We help user to Master core Product Management Skills and AI workflows to stay relevant in the next wave of Product Careers.",
            "messages" : []
        }

        with st.spinner("Analyzing company information..."):
            try:
                result = full_agent.invoke(input_state, config=RunnableConfig())
                final_output = result.get("answer") or result.get("output") or result
                st.success("✅ Answer:")
                st.write(final_output)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
