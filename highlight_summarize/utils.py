import os
import sys
import json
import time
import logging
from typing import Any
from pydantic import BaseModel
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
BATCH_POLL_INTERVAL = 30  # seconds

load_dotenv()
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")


# We initialize every time to make the most of HF's caching.
def openai_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_version="2025-01-01-preview",
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


def create_batch_request(
    custom_id: str,
    messages: list[dict[str, str]],
    model_name: str,
    temperature: float,
    response_format: type[BaseModel] | None = None,
) -> dict[str, Any]:
    """Create a single batch request in the format expected by Azure OpenAI Batch API."""
    body = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        schema = response_format.model_json_schema()
        # Azure OpenAI requires additionalProperties: false for strict mode
        schema["additionalProperties"] = False
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "strict": True,
                "schema": schema,
            },
        }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/chat/completions",
        "body": body,
    }


def submit_batch(batch_file_path: str) -> str:
    """Upload a JSONL batch file and submit a batch job. Returns the batch job ID."""
    with open(batch_file_path, "rb") as f:
        upload_response = client.files.create(file=f, purpose="batch")

    batch_job = client.batches.create(
        input_file_id=upload_response.id,
        endpoint="/v1/chat/completions",  # type: ignore[arg-type]
        completion_window="24h",
    )
    print(f"Batch job submitted: {batch_job.id}")
    return batch_job.id


def poll_batch(batch_job_id: str) -> str | None:
    """Poll until the batch job completes. Returns the output file ID."""
    start_time = time.time()
    while True:
        batch_job = client.batches.retrieve(batch_job_id)
        status = batch_job.status
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)
        print(f"\rBatch {batch_job_id} status: {status} [{mins:02d}:{secs:02d}]", end="", flush=True)

        if status == "completed":
            print()  # Move to new line
            return batch_job.output_file_id
        elif status in ("failed", "expired", "cancelled"):
            print()  # Move to new line
            error_file_id = batch_job.error_file_id
            if error_file_id:
                errors = client.files.content(error_file_id).content.decode("utf-8")
                raise RuntimeError(
                    f"Batch job {batch_job_id} {status}. Errors:\n{errors}"
                )
            raise RuntimeError(f"Batch job {batch_job_id} {status}")

        time.sleep(BATCH_POLL_INTERVAL)


def download_batch_results(output_file_id: str) -> list[dict[str, Any]]:
    """Download and parse batch results. Returns list of {custom_id, response_body} dicts."""
    content = client.files.content(output_file_id).content.decode("utf-8")
    results = []
    for line in content.strip().split("\n"):
        if not line:
            continue
        result = json.loads(line)
        results.append(
            {
                "custom_id": result["custom_id"],
                "response_body": result["response"]["body"],
            }
        )
    return results


def run_batch(
    requests: list[dict[str, Any]],
    batch_file_path: str,
    cleanup: bool = True,
) -> list[dict[str, Any]]:
    """
    Run a batch of requests through Azure OpenAI Batch API.

    Args:
        requests: List of batch request dicts (from create_batch_request)
        batch_file_path: Path to write the JSONL file
        cleanup: Whether to delete local and remote files after completion

    Returns:
        List of results sorted by custom_id (assuming numeric IDs)
    """
    # Write JSONL file
    with open(batch_file_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    # Submit, poll, download
    batch_job_id = submit_batch(batch_file_path)
    output_file_id = poll_batch(batch_job_id)
    if output_file_id is None:
        raise RuntimeError(f"Batch job {batch_job_id} completed but no output file ID")
    results = download_batch_results(output_file_id)

    # Cleanup local batch file
    if cleanup:
        try:
            os.remove(batch_file_path)
            logger.info(f"Deleted local batch file {batch_file_path}")
        except OSError as e:
            logger.warning(f"Failed to delete local batch file {batch_file_path}: {e}")

    return results
