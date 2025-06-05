from datasets import load_dataset, concatenate_datasets

from .utils import NOANSWER_PRED


def load_repliqa(split=3):
    dataset_name = f"repliqa_{split}"
    print(f"Loading dataset: {dataset_name}")
    repliqa = load_dataset("ServiceNow/repliqa")[dataset_name]

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

def load_bioasq():
    print("Loading dataset: bioasq")
    dataset_qa_traintest = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages")
    dataset_qa = concatenate_datasets([
        dataset_qa_traintest["train"],
        dataset_qa_traintest["test"]
    ])
    dataset_corpus = load_dataset("enelpol/rag-mini-bioasq", "text-corpus")["test"]

    # Corpus to dict for speed.
    corpus_dict = {}
    for id, passage in zip(dataset_corpus["id"], dataset_corpus["passage"]):
        if id in corpus_dict:
            raise ValueError(f"Duplicate id {id} in corpus.")
        corpus_dict[id] = passage

    # Replace the relevant passage ids with the actual passages.
    def include_passages(example):
        passages = []
        for id in example["relevant_passage_ids"]:
            passages.append(corpus_dict[id].replace("\n", " "))

        return {"document_extracted": "\n\n".join(passages)}

    return dataset_qa.map(include_passages, remove_columns=["relevant_passage_ids"])