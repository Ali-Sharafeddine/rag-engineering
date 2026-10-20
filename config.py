import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


KNOWLEDGE_PATH = BASE_DIR / "documents" / "knowledge.txt"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

LLM_MODEL = "gpt-4.1-mini"

TOP_K = 3

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MIN_RETRIEVAL_SCORE = 0.25