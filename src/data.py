import datasets

from .utils import NOANSWER_PRED

def load_repliqa(split=3):
    dataset_name = f"repliqa_{split}"
    print(f"Loading dataset: {dataset_name}")
    repliqa = datasets.load_dataset("ServiceNow/repliqa")[dataset_name]

    # Fix labels in the dataset.
    def fix_label(example):
        if example["answer"] == "The answer is not found in the document.":
            return {"answer": NOANSWER_PRED}

        # Fix wrongly labelled.
        wrongly_labelled = [
            "The title of the mural is not mentioned in the document.",
            "Clara Bennett's book release date is not provided in the document.",
            "The document doesn't specify the exact date, it only mentions the 'late 2010s.'"
        ]
        if "The document does not" in example["answer"] or example["answer"] in wrongly_labelled:
            return {"answer": NOANSWER_PRED}
        
        return example

    clean_repliqa = repliqa.map(fix_label)

    return clean_repliqa