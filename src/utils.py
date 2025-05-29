import os
from openai import AzureOpenAI
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# We initialize every time to make the most of HF's caching.
def openai_client():
    # return AzureOpenAI(
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    #     api_version="2024-12-01-preview",
    #     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    # )
    return AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=endpoint,
        azure_ad_token_provider=get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"),
    )