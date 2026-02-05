import os
import yaml
import time
import datasets
from typing import Any

from highlight_summarize.qa import QAEvaluator
from highlight_summarize.data import load_dataset
from highlight_summarize.judges import JUDGES_MAP
from highlight_summarize.threads import mt_map
from highlight_summarize.hs import (
    HSBaseline,
    HSStructuredHighlighter,
    HSBERTExtractor,
    HSSpanHighlighter,
    HSTwoStepsHighlighter,
)


PIPELINE_MAP = {
    "QAEvaluator": QAEvaluator,
    "HSBaseline": HSBaseline,
    "HSStructuredHighlighter": HSStructuredHighlighter,
    "HSSpanHighlighter": HSSpanHighlighter,
    "HSBERTExtractor": HSBERTExtractor,
    "HSTwoStepsHighlighter": HSTwoStepsHighlighter,
}


def run_id(experiment_config, dataset_name):
    """Generate a unique run ID for the experiment based on its configuration and dataset name.
    The run ID is also the directory where the results will be stored.
    """
    if experiment_config["pipeline"] == "QAEvaluator":
        return (
            f"{experiment_config['results_dir']}/"
            f"{dataset_name}/{experiment_config['pipeline']}-{experiment_config['model_name']}"
        )

    return (
        f"{experiment_config['results_dir']}/"
        f"{dataset_name}/{experiment_config['pipeline']}"
        f"-{experiment_config['highlighter_model_name'].replace('/', '_')}"
        f"-{experiment_config['summarizer_model_name']}"
    )


def load_config(path):
    """Load the experiment configuration from a YAML file."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    # Experiments are repeated for each dataset.
    experiments = {}
    for dataset_name in config["datasets"]:
        for experiment_config in config["experiments"].values():
            ex = experiment_config.copy()
            ex["dataset"] = dataset_name
            experiments[run_id(ex, dataset_name)] = ex

    config["experiments"] = experiments

    return config


def run_inference(run_id, experiment_config, max_threads) -> datasets.Dataset:
    """Run the inference for a given experiment configuration.
    This function would typically call the actual inference logic.
    """
    dst_dir = f"{run_id}/inference"
    # Try to load the existing results.
    if os.path.exists(dst_dir):
        prediction_dataset = datasets.load_from_disk(dst_dir)
        if "answer_pred" in prediction_dataset.column_names:
            print(f"    > Skipping inference for {run_id} as results already exist.")
            return prediction_dataset
        else:
            raise Exception(
                f"{run_id}: This shouldn't happen: missing 'answer_pred' column but dataset exists."
            )

    print(f"    > Running inference for {run_id}.")
    print(f"    > Loading dataset {experiment_config['dataset']}.")
    dataset = load_dataset(experiment_config["dataset"])

    # Set up the pipeline.
    pipeline_cls = PIPELINE_MAP.get(experiment_config["pipeline"])
    if pipeline_cls is None:
        raise ValueError(f"Pipeline {experiment_config['pipeline']} is not supported.")
    if experiment_config["pipeline"] == "QAEvaluator":
        pipeline = pipeline_cls(
            model_name=experiment_config["model_name"],
            temperature=experiment_config["temperature"],
        )
    else:
        pipeline = pipeline_cls(
            highlighter_model_name=experiment_config["highlighter_model_name"],
            summarizer_model_name=experiment_config["summarizer_model_name"],
            temperature=experiment_config["temperature"],
        )

    # Run.
    now = time.time()
    prediction_dataset = mt_map(
        function=pipeline,
        dataset=dataset,
        max_threads=max_threads,
    )
    elapsed_time = time.time() - now

    with open(f"{run_id}/run.txt", "w") as f:
        f.write(
            f"Elapsed time for predictions: {elapsed_time:.2f} seconds. Average: {elapsed_time / len(dataset):.2f} seconds per example."
        )

    # Store.
    print(f"    > Storing predictions to {dst_dir}.")
    prediction_dataset.save_to_disk(dst_dir)

    return prediction_dataset


def run_judgement(
    run_id: str,
    prediction_dataset: datasets.Dataset,
    judges_config: dict,
    max_threads: int,
) -> datasets.Dataset:
    """Run the judgement for a given experiment configuration.
    This function would typically call the actual judgement logic.
    """
    judges = judges_config["judges"]
    dst_dir = f"{run_id}/judgement"
    # Try to load the existing results.
    if os.path.exists(dst_dir):
        print(
            f"    > Skipping judgement for {run_id} as results already exist in {dst_dir}."
        )
        judged_dataset = datasets.load_from_disk(dst_dir)
        # Check that all judges are present.
        for judge_name in judges:
            # We just string-match for simplicity.
            for column_name in judged_dataset.column_names:
                if judge_name in column_name:
                    break
            else:
                raise ValueError(
                    f"Judge {judge_name} is missing from the dataset {run_id}. Delete the directory to rerun the judgement."
                )
        return judged_dataset

    # Run.
    judged_dataset = prediction_dataset
    for judge_name in judges:
        judge_cls = JUDGES_MAP.get(judge_name)
        if judge_cls is None:
            raise ValueError(f"Judge {judge_name} is not supported.")
        print(f"    > Running judgement for {run_id} with judge {judge_name}.")
        judge = judge_cls(
            model_name=judges_config["model_name"],
        )

        judged_dataset = mt_map(
            function=judge,
            dataset=judged_dataset,
            max_threads=max_threads,
        )
    # Store.
    print(f"    > Storing judged predictions to {dst_dir}.")
    judged_dataset.save_to_disk(dst_dir)

    return judged_dataset


def load_all_results(results_dir="results/") -> dict[str, datasets.Dataset]:
    """Load all results from the results directory and combine them by dataset."""
    combined_results = {}

    for dataset_name in os.listdir(results_dir):
        if not os.path.isdir(os.path.join(results_dir, dataset_name)):
            continue
        print(f"Processing dataset: {dataset_name}")

        dataset_results = []
        for run_id in os.listdir(os.path.join(results_dir, dataset_name)):
            if not os.path.isdir(os.path.join(results_dir, dataset_name, run_id)):
                continue

            dirname = os.path.join(results_dir, dataset_name, run_id, "judgement")
            if not os.path.isdir(dirname):
                print(f"Skipping {run_id} as {dirname} doesn't exist.")
                continue

            try:
                res = datasets.load_from_disk(dirname)
                # Add run_id and pipeline columns to each row
            except Exception as e:
                print(f"Error loading {run_id} from {dirname}: {e}")
                continue
            # FIXME: this assumes that the pipeline name doesn't have dashes.
            pipeline = run_id.split("-")[0]
            res = res.add_column("run_id", [run_id] * len(res))
            res = res.add_column("pipeline", [pipeline] * len(res))
            dataset_results.append(res)

        # Concatenate all results for this dataset
        if dataset_results:
            combined_results[dataset_name] = datasets.concatenate_datasets(
                dataset_results
            )

    return combined_results


if __name__ == "__main__":
    config = load_config("experiments.yaml")

    for run_id, experiment_config in config["experiments"].items():
        os.makedirs(run_id, exist_ok=True)
        print(f"[*] Running experiment: {run_id}")
        # Store the configuration in the run directory.
        if not os.path.exists(f"{run_id}/experiment.yaml"):
            with open(f"{run_id}/experiment.yaml", "w") as f:
                yaml.dump(experiment_config, f)
        else:
            # Check if the experiment config has changed.
            with open(f"{run_id}/experiment.yaml", "r") as f:
                existing_config = yaml.safe_load(f)
            if existing_config != experiment_config:
                raise ValueError(
                    f"Experiment configuration for {run_id} has changed. "
                    "Please remove the existing directory to rerun the experiment."
                )

        max_threads = config.get("max_threads", 8)
        prediction_dataset = run_inference(run_id, experiment_config, max_threads)
        run_judgement(run_id, prediction_dataset, config["judges_config"], max_threads)
