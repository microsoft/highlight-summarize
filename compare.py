"""Compares between two pipelines (run_id).

The `highlighter` mode compares between a highlighter and a full HS pipeline.
The `pairwise` command compares the pairwise answers produced by two runs,
for all pipelines.

Usage:
    compare.py highlighter <run_folder>
    compare.py pairwise <results_for_dataset_folder>

A `run_folder` is produced by `run_experiments.py` and it looks something like:
`results/repliqa_3/HSBaseline-gpt-4.1-mini-gpt-4.1-mini`
where `results/` is the base folder for the results, `repliqa_3` is the dataset name
and `HSBaseline-gpt-4.1-mini-gpt-4.1-mini` is the pipeline.

The `results_for_dataset_folder` is the folder containing the results for a specific dataset,
e.g. `results/repliqa_3`.
"""
import os
import datasets
import concurrent
import pandas as pd
from tqdm import tqdm
from docopt import docopt

from src.comparison_judge import ComparisonJudge

def get_run_info(run_folder):
    """Extracts the base folder, dataset name and pipeline from the run folder."""
    parts = run_folder.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError("Invalid run folder format.")
    base_folder = os.path.join(*parts[:-2])
    dataset_name = parts[-2]
    pipeline = parts[-1]
    if not base_folder or not dataset_name or not pipeline:
        raise ValueError(f"This shouldn't happen. Base folder: {base_folder}, dataset name: {dataset_name}, pipeline: {pipeline}")
    return base_folder, dataset_name, pipeline

def load_from_run(run_folder):
    """Loads the inference from the run folder."""
    if not os.path.exists(run_folder):
            raise ValueError(f"Run folder does not exist: {run_folder}")
    if not os.path.exists(os.path.join(run_folder, "inference")):
        raise ValueError(f"Run folder does not contain 'inference' directory: {run_folder}. You should run `run_experiments.py` first.")
    return datasets.load_from_disk(os.path.join(run_folder, "inference"))

def pairwise_comparison(run_folder_1, run_folder_2):
    """Compares the pairwise answers produced by two runs."""
    # Ensure that we're comparing the right things.
    base_folder_1, dataset_name_1, pipeline_1 = get_run_info(run_folder_1)
    base_folder_2, dataset_name_2, pipeline_2 = get_run_info(run_folder_2)
    if base_folder_1 != base_folder_2:
        raise ValueError("Base folders do not match: {} vs {}".format(base_folder_1, base_folder_2))
    if dataset_name_1 != dataset_name_2:
        raise ValueError("Datasets do not match: {} vs {}".format(dataset_name_1, dataset_name_2))
    if pipeline_1 == pipeline_2:
        raise ValueError("Pipelines must be different: {} vs {}".format(pipeline_1, pipeline_2))

    # Load the datasets from the run folders.
    dataset_1 = load_from_run(run_folder_1)
    dataset_2 = load_from_run(run_folder_2)

    # Comparison.
    judge = ComparisonJudge()
    compared = []
    for example1, example2 in tqdm(zip(dataset_1, dataset_2), total=len(dataset_1)):
        question = example1["question"]
        if question != example2["question"]:
            print(f"Questions do not match: {question} vs {example2['question']}")
            continue
        output_1 = example1["answer_pred"]
        output_2 = example2["answer_pred"]
        result = judge(question, output_1, output_2)
        compared.append({
            "dataset_name": dataset_name_1,
            "question": question,
            f"{pipeline_1}-output": output_1,
            f"{pipeline_2}-output": output_2,
            "preference": pipeline_1 if result.response_id == 1 else pipeline_2,
            "explanation": result.explanation,
        })
    fname = os.path.join(base_folder_1, dataset_name_1, f'comparison-{pipeline_1}_vs_{pipeline_2}.jsonl')
    print(f"Saving comparison results to {fname}")
    pd.DataFrame(compared).to_json(fname, lines=True, orient="records")
      

if __name__ == "__main__":
    args = docopt(__doc__)
    if args["pairwise"]:
        folder = args["<results_for_dataset_folder>"]
        if not os.path.exists(folder):
            raise ValueError(f"Results folder does not exist: {folder}")
        pipelines = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        if len(pipelines) < 2:
            raise ValueError(f"Not enough pipelines found in the folder: {folder}. Found: {pipelines}")
        
        # Run the comparisons, one thread each.
        print(f"Found {len(pipelines)} pipelines in folder '{folder}'.")
        max_threads = len(pipelines) * (len(pipelines) - 1) // 2
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            to_run = []
            for i in range(len(pipelines)):
                for j in range(i + 1, len(pipelines)):
                    run_folder_1 = os.path.join(folder, pipelines[i])
                    run_folder_2 = os.path.join(folder, pipelines[j])
                    to_run.append((run_folder_1, run_folder_2))
            print(f"Running {len(to_run)} pairwise comparisons.")
            futures = [executor.submit(pairwise_comparison, run_folder_1, run_folder_2) for run_folder_1, run_folder_2 in to_run]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in pairwise comparison: {e}")
    elif args["highlighter"]:
        judge = ComparisonJudge()
        run_folder = args["<run_folder>"]
        base_folder, dataset_name, pipeline = get_run_info(run_folder)
        print(f"Comparing highlighter output for dataset '{dataset_name}' and pipeline '{pipeline}'.")
        dataset = load_from_run(run_folder)

        if not "highlighter_extracted" in dataset.column_names:
            raise ValueError("Dataset does not contain 'highlighter_extracted' column.")

        compared = []

        for example in tqdm(dataset):
            result = judge(example["question"], example["answer_pred"], example["highlighter_extracted"])
            compared.append({
                "dataset_name": dataset_name,
                "pipeline": pipeline,
                "question": example["question"],
                "hs-output": example["answer_pred"],
                "highlighter-output": example["highlighter_extracted"],
                "preference": "hs" if result.response_id == 1 else "highlighter",
                "explanation": result.explanation,
            })
        fname = os.path.join(base_folder, dataset_name, f'comparison-{pipeline}-highlighter_vs_hs.jsonl')
        print(f"Saving comparison results to {fname}")
        pd.DataFrame(compared).to_json(fname, lines=True, orient="records")
    else:
        print("Invalid command. Use 'answers' or 'highlighter'.")