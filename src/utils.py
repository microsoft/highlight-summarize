import os
import sys
import logging
from openai import AzureOpenAI, NOT_GIVEN
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)
logger = logging.getLogger(__name__)

# COMMON.
OPENAI_TIMEOUT = 20
OPENAI_RETRIES = 5
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
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        ),
        timeout=OPENAI_TIMEOUT,
        max_retries=OPENAI_RETRIES,
    )


client = openai_client()


def query_llm(
    messages: list[dict[str, str]], temperature, model_name: str, response_format=None
):
    # Structured output: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python-secure%2Cdotnet-entra-id&pivots=programming-language-python.
    model_response = client.beta.chat.completions.parse(
        messages=messages,
        temperature=temperature,
        model=model_name,
        response_format=response_format or NOT_GIVEN,
    )

    if (
        not model_response
        or not model_response.choices
        or not model_response.choices[0]
    ):
        return None

    if response_format is None:
        return model_response.choices[0].message.content

    return model_response.choices[0].message.parsed
