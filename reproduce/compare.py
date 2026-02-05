"""Compares between two pipelines (run_id) using Azure OpenAI Batch API.

The `highlighter` mode compares between a highlighter and a full HS pipeline.
The `pairwise` command compares the pairwise answers produced by two runs,
for all pipelines.

Usage:
    compare.py highlighter <run_folder> [--model=<model>]
    compare.py pairwise <results_for_dataset_folder> [--model=<model>]

Options:
    --model=<model>  Model name for the judge [default: gpt-4.1-mini-batch]

A `run_folder` is produced by `run_experiments.py` and it looks something like:
`results/repliqa_3/HSBaseline-gpt-4.1-mini-gpt-4.1-mini`
where `results/` is the base folder for the results, `repliqa_3` is the dataset name
and `HSBaseline-gpt-4.1-mini-gpt-4.1-mini` is the pipeline.

The `results_for_dataset_folder` is the folder containing the results for a specific dataset,
e.g. `results/repliqa_3`.
"""

import os
import datasets
import pandas as pd
from docopt import docopt

from highlight_summarize.comparison_judge import (
    ComparisonJudge,
    ResponseChoice,
    JudgeResponse,
)
from highlight_summarize.utils import run_batch


def get_run_info(run_folder):
    """Extracts the base folder, dataset name and pipeline from the run folder."""
    parts = run_folder.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid run folder format.")
    base_folder = os.path.join(*parts[:-2])
    dataset_name = parts[-2]
    pipeline = parts[-1]
    if not base_folder or not dataset_name or not pipeline:
        raise ValueError(
            f"This shouldn't happen. Base folder: {base_folder}, dataset name: {dataset_name}, pipeline: {pipeline}"
        )
    return base_folder, dataset_name, pipeline


def load_from_run(run_folder):
    """Loads the inference from the run folder."""
    if not os.path.exists(run_folder):
        raise ValueError(f"Run folder does not exist: {run_folder}")
    if not os.path.exists(os.path.join(run_folder, "inference")):
        raise ValueError(
            f"Run folder does not contain 'inference' directory: {run_folder}. You should run `run_experiments.py` first."
        )
    return datasets.load_from_disk(os.path.join(run_folder, "inference"))


def parse_judge_preference(
    response: JudgeResponse, option_1: str, option_2: str
) -> str:
    """Parses the judge's response and returns the preferred option."""
    if response.preference == ResponseChoice.response_1:
        return option_1
    elif response.preference == ResponseChoice.response_2:
        return option_2
    elif response.preference == ResponseChoice.tie:
        return "tie"
    elif response.preference == ResponseChoice.neither:
        return "neither"
    else:
        return "Error: unknown preference"


def pairwise_comparison(run_folder_1, run_folder_2, model_name="gpt-4.1-mini"):
    """Compares the pairwise answers produced by two runs using batch API."""
    # Ensure that we're comparing the right things.
    base_folder_1, dataset_name_1, pipeline_1 = get_run_info(run_folder_1)
    base_folder_2, dataset_name_2, pipeline_2 = get_run_info(run_folder_2)
    if base_folder_1 != base_folder_2:
        raise ValueError(
            "Base folders do not match: {} vs {}".format(base_folder_1, base_folder_2)
        )
    if dataset_name_1 != dataset_name_2:
        raise ValueError(
            "Datasets do not match: {} vs {}".format(dataset_name_1, dataset_name_2)
        )
    if pipeline_1 == pipeline_2:
        raise ValueError(
            "Pipelines must be different: {} vs {}".format(pipeline_1, pipeline_2)
        )

    output_fname = os.path.join(
        base_folder_1, dataset_name_1, f"comparison-{pipeline_1}_vs_{pipeline_2}.jsonl"
    )
    if os.path.exists(output_fname):
        print(f"Comparison file already exists: {output_fname}. Skipping.")
        return

    # Load the datasets from the run folders.
    dataset_1 = load_from_run(run_folder_1)
    dataset_2 = load_from_run(run_folder_2)

    # Collect all comparisons and create batch requests.
    judge = ComparisonJudge(model_name=model_name)
    batch_requests = []
    metadata = []  # Store metadata for each comparison

    for idx, (example1, example2) in enumerate(zip(dataset_1, dataset_2)):
        question = example1["question"]
        if question != example2["question"]:
            print(
                f"Questions do not match at index {idx}: {question} vs {example2['question']}"
            )
            continue

        output_1 = example1["answer_pred"]
        output_2 = example2["answer_pred"]

        request, _ = judge.create_batch_request(
            custom_id=str(idx),
            question=question,
            output_1=output_1,
            output_2=output_2,
        )
        batch_requests.append(request)
        metadata.append(
            {
                "idx": idx,
                "question": question,
                "output_1": output_1,
                "output_2": output_2,
            }
        )

    if not batch_requests:
        print("No valid comparisons to process.")
        return

    # Run batch
    batch_file = os.path.join(
        base_folder_1, dataset_name_1, f"batch-{pipeline_1}_vs_{pipeline_2}.jsonl"
    )
    print(f"Submitting batch with {len(batch_requests)} comparisons...")
    results = run_batch(batch_requests, batch_file)

    # Parse results and build output
    results_by_id = {}
    for result in results:
        original_id, judge_response = ComparisonJudge.parse_batch_response(result)
        results_by_id[original_id] = judge_response

    compared = []
    for meta in metadata:
        idx_str = str(meta["idx"])
        if idx_str not in results_by_id:
            print(f"Missing result for index {idx_str}")
            continue

        response = results_by_id[idx_str]
        compared.append(
            {
                "dataset_name": dataset_name_1,
                "question": meta["question"],
                f"{pipeline_1}-output": meta["output_1"],
                f"{pipeline_2}-output": meta["output_2"],
                "preference": parse_judge_preference(response, pipeline_1, pipeline_2),
                "explanation": response.explanation,
            }
        )

    print(f"Saving comparison results to {output_fname}")
    pd.DataFrame(compared).to_json(output_fname, lines=True, orient="records")


def highlighter_comparison(run_folder, model_name="gpt-4.1-mini", limit_words=40):
    """Compares highlighter output vs HS pipeline output using batch API."""
    base_folder, dataset_name, pipeline = get_run_info(run_folder)
    print(
        f"Comparing highlighter output for dataset '{dataset_name}' and pipeline '{pipeline}'."
    )

    output_fname = os.path.join(
        base_folder, dataset_name, f"comparison-{pipeline}-highlighter_vs_hs.jsonl"
    )
    if os.path.exists(output_fname):
        print(f"Comparison file already exists: {output_fname}. Skipping.")
        return

    dataset = load_from_run(run_folder)

    if "highlighter_extracted" not in dataset.column_names:
        raise ValueError("Dataset does not contain 'highlighter_extracted' column.")

    # Collect all comparisons and create batch requests.
    judge = ComparisonJudge(model_name=model_name)
    batch_requests = []
    metadata = []

    for idx, example in enumerate(dataset):
        question = example["question"]
        hs_output = example["answer_pred"]
        highlighter_output = example["highlighter_extracted"]

        # Truncate highlighter output to limit_words
        def truncate(text, limit_words):
            if limit_words is None or text is None:
                return text
            words = text.split()
            if len(words) <= limit_words:
                return text
            return " ".join(words[:limit_words]) + " ..."

        hs_output = truncate(hs_output, limit_words)
        highlighter_output = truncate(highlighter_output, limit_words)

        request, _ = judge.create_batch_request(
            custom_id=str(idx),
            question=question,
            output_1=hs_output,
            output_2=highlighter_output,
        )
        batch_requests.append(request)
        metadata.append(
            {
                "idx": idx,
                "question": question,
                "hs_output": hs_output,
                "highlighter_output": highlighter_output,
            }
        )

    if not batch_requests:
        print("No valid comparisons to process.")
        return

    # Run batch
    batch_file = os.path.join(
        base_folder, dataset_name, f"batch-{pipeline}-highlighter_vs_hs.jsonl"
    )
    print(f"Submitting batch with {len(batch_requests)} comparisons...")
    results = run_batch(batch_requests, batch_file)

    # Parse results and build output
    results_by_id = {}
    for result in results:
        original_id, judge_response = ComparisonJudge.parse_batch_response(result)
        results_by_id[original_id] = judge_response

    compared = []
    for meta in metadata:
        idx_str = str(meta["idx"])
        if idx_str not in results_by_id:
            print(f"Missing result for index {idx_str}")
            continue

        response = results_by_id[idx_str]
        compared.append(
            {
                "dataset_name": dataset_name,
                "pipeline": pipeline,
                "question": meta["question"],
                "hs-output": meta["hs_output"],
                "highlighter-output": meta["highlighter_output"],
                "preference": parse_judge_preference(response, "hs", "highlighter"),
                "explanation": response.explanation,
            }
        )

    print(f"Saving comparison results to {output_fname}")
    pd.DataFrame(compared).to_json(output_fname, lines=True, orient="records")


if __name__ == "__main__":
    args = docopt(__doc__)
    model_name = args["--model"]

    if args["pairwise"]:
        folder = args["<results_for_dataset_folder>"]
        if not os.path.exists(folder):
            raise ValueError(f"Results folder does not exist: {folder}")
        pipelines = [
            d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))
        ]
        if len(pipelines) < 2:
            raise ValueError(
                f"Not enough pipelines found in the folder: {folder}. Found: {pipelines}"
            )

        print(f"Found {len(pipelines)} pipelines in folder '{folder}'.")

        # Process pipeline pairs sequentially (batch API handles parallelism)
        for i in range(len(pipelines)):
            for j in range(i + 1, len(pipelines)):
                run_folder_1 = os.path.join(folder, pipelines[i])
                run_folder_2 = os.path.join(folder, pipelines[j])
                print(f"\n--- Comparing {pipelines[i]} vs {pipelines[j]} ---")
                try:
                    pairwise_comparison(
                        run_folder_1, run_folder_2, model_name=model_name
                    )
                except Exception as e:
                    print(f"Error in pairwise comparison: {e}")

    elif args["highlighter"]:
        run_folder = args["<run_folder>"]
        highlighter_comparison(run_folder, model_name=model_name)

    else:
        print("Invalid command. Use 'pairwise' or 'highlighter'.")
