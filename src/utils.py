import os
from typing import Literal
from openai import AzureOpenAI, NOT_GIVEN
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# COMMON.
NOANSWER_PRED = "UNANSWERABLE"
FAILED_PRED = "FAILED"

load_dotenv()
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

# We initialize every time to make the most of HF's caching.
def openai_client() -> AzureOpenAI:
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

@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(5),
)
def query_llm(messages: list[dict[str, str]], temperature, model_name: str, response_format = None):
    # Structured output: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python-secure%2Cdotnet-entra-id&pivots=programming-language-python.
    model_response = openai_client().beta.chat.completions.parse(
        messages=messages,
        temperature=temperature,
        model=model_name,
        response_format=response_format or NOT_GIVEN,
    )

    if not model_response or not model_response.choices or not model_response.choices[0]:
        return None

    if response_format is None:
        return model_response.choices[0].message.content

    return model_response.choices[0].message.parsed