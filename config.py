import os
import streamlit as st

# Handle both local (.env) and cloud (secrets) configurations
try:
    # Try Streamlit secrets first (for cloud deployment)
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    # Fallback to environment variables (for local development)
    from dotenv import load_dotenv
    load_dotenv()
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL_NAME = "llama3-8b-8192"