import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

# We initialize every time to make the most of HF's caching.
def openai_client():
    return AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )