"""Utility functions for the reproduce module."""
import os
import datasets


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
