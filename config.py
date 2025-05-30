# config.py
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "llama3-8b-8192" # A good balance of speed and capability