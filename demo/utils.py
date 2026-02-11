import os
import json
import streamlit as st

CONTENTS_DIR = "contents"
LOG_FILE = os.getenv("LOG_FILE", "./chats_history.jsonl")

# Logging.
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(data: dict):
    """Log the interaction to a JSONL file."""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')

def load_content(fname, mode="r"):
    """Load content from a file in the contents directory."""
    if mode == 'r':
        errors = 'ignore'
        encoding = 'utf-8'
    else:
        errors = None
        encoding = None

    with open(os.path.join(CONTENTS_DIR, fname), mode, errors=errors, encoding=encoding) as f:
        return f.read()

def sidebar():
    with st.sidebar:
        st.title("Highlight & Summarize")
        st.markdown("A RAG design pattern for secure, high-quality LLM-based question answering.")
        st.markdown("📖 [Read the paper](https://arxiv.org/abs/2508.02872)")
        st.markdown('<img src="https://pngimg.com/d/github_PNG40.png" width="30"> [Code](https://github.com/microsoft/highlight-summarize)', unsafe_allow_html=True)