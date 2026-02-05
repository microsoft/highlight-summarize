"""Convert HuggingFace Dataset results to CSV, stripping HF dataset columns.
The merge key (question_id for repliqa, id for bioasq) is preserved for later re-merging.
"""
import os
from run_experiments import load_all_results

RESULTS_DIR = "results/"

# Columns from HuggingFace datasets to drop (keep only evaluation results + merge key)
HF_COLUMNS = {
    "repliqa": ["document_id", "document_topic", "document_path", "document_extracted", 
                "question", "answer", "long_answer"],
    "bioasq": ["question", "answer", "document_extracted"],
}

results = load_all_results(RESULTS_DIR)
for dataset_name, dataset_results in results.items():
    print(f"Processing: {dataset_name}")
    df = dataset_results.to_pandas()
    
    # Determine which HF columns to drop based on dataset type
    hf_cols = HF_COLUMNS.get("repliqa" if "repliqa" in dataset_name else "bioasq", [])
    df = df.drop(columns=[c for c in hf_cols if c in df.columns])
    
    df.to_csv(os.path.join(RESULTS_DIR, dataset_name, f"{dataset_name}_results_processed.csv"), index=False)
