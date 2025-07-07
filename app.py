# Streamlit UI for user input and chatbot response display:
import streamlit as st
from inference import ask_question

# Suppress Streamlit warnings and disable usage tracking:
import os
os.environ["STREAMLIT_SUPPRESS_CONFIG_WARNINGS"] = "1"
os.environ["STREAMLIT_DISABLE_USAGE_STATS"] = "true"

# Setting up the Streamlit page and title:
st.set_page_config(page_title="CRAG Chatbot", layout="wide")
st.title("🫀 CRAG: Cardiology RAG Chatbot")

# Display the welcome message and instructions:
st.markdown("""
Welcome to the **CRAG Medical Assistant** 👩‍⚕️

Ask any question related to heart health, diagnosis, treatment, or lifestyle — based on trusted documents and web search fallback.
""")

# Display input field for heart-related question:
query = st.text_input("Ask a heart-related question:", placeholder="e.g. How does high blood pressure affect the heart?", key="input")

# Handle query submission and display generated response:
if st.button("Ask") and query:
    with st.spinner("Generating answer..."):
        try:
            print(f"Received query: {query}")
            answer = ask_question(query)
            st.success("Answer:")
            st.markdown(f"{answer}")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")
