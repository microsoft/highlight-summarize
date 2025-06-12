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


def query_llm(
    messages: list[dict[str, str]], temperature, model_name: str, response_format=None
):
    # We create a new client for each query. While not ideal, this
    # avoids some connection leak issues that have been occurring.
    with AzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint=endpoint,
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
        ),
        timeout=OPENAI_TIMEOUT,
        max_retries=OPENAI_RETRIES,
    ) as client:
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