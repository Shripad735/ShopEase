import os
import streamlit as st

# Try Streamlit secrets first, then environment variables
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL_NAME = "llama3-8b-8192"