"""Fine-tune a question-answering model on the RepliQA dataset."

Usage:
    qa-extractor-finetune.py [--model=<model_name>]

Options:
    --model=<model_name>  Name of the model to fine-tune [default: deepset/deberta-v3-base-squad2]
"""

import os
from docopt import docopt
from datasets import concatenate_datasets
from transformers import (
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    DefaultDataCollator,
)

from highlight_summarize.data import load_repliqa


def run(model_name):
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess the RepliQA dataset.
    NOANSWER = "UNANSWERABLE"
    repliqa_splits = [
        load_repliqa(split=i) for i in range(3)
    ]  # We exclude the test set (split 3).
    repliqa = concatenate_datasets(repliqa_splits)

    def filter_bad_answers(example):
        if example["answer"] != NOANSWER:
            return (
                example["long_answer"].lower() in example["document_extracted"].lower()
            )
        return True

    repliqa_filtered = repliqa.filter(filter_bad_answers)
    print(f"Filtered dataset size: {len(repliqa_filtered)}")
    print(
        f"Unanswerable examples: {len(repliqa_filtered.filter(lambda x: x['answer'] == NOANSWER))}"
    )

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["document_extracted"],
            max_length=1600,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
            # padding="longest",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answer"]
        long_answers = examples["long_answer"]
        docs = examples["document_extracted"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            if answers[i] == NOANSWER:
                start_positions.append(0)
                end_positions.append(0)
                continue

            long_answer = long_answers[i].lower()
            doc = docs[i].lower()
            sequence_ids = inputs.sequence_ids(i)

            start_char = doc.find(long_answer)
            if start_char != -1:
                end_char = start_char + len(long_answer)

                # Find the start and end of the context.
                idx = 0
                while sequence_ids[idx] != 1:
                    idx += 1
                context_start = idx
                while sequence_ids[idx] == 1:
                    idx += 1
                context_end = idx - 1

                # If the answer is not fully inside the context.
                if (
                    offset[context_start][0] > end_char
                    or offset[context_end][1] < start_char
                ):
                    print(f"Answer: {long_answer}.")
                    print(f"Document: {doc}.")
                    print(f"Context start: {context_start}, end: {context_end}.")
                    print(f"Offset: {offset}.")
                    print(f"Start char: {start_char}, end char: {end_char}.")
                    print(f"Sequence ids: {sequence_ids}.")
                    raise ValueError("The answer is not fully inside the `max_length`.")
                else:
                    # Otherwise it's the start and end token positions.
                    idx = context_start
                    while idx <= context_end and offset[idx][0] <= start_char:
                        idx += 1
                    start_positions.append(idx - 1)

                    idx = context_end
                    while idx >= context_start and offset[idx][1] >= end_char:
                        idx -= 1
                    end_positions.append(idx + 1)
            else:
                raise ValueError(
                    "This shouldn't happen: you must filter out "
                    "`long_answer`s that aren't exactly in the "
                    "context document.\n"
                    f" Answer: {long_answer}.\nDocument: {doc}"
                )

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    # Tokenize the dataset.
    tokenized_repliqa = repliqa_filtered.map(preprocess_function, batched=True)

    # Train-test split.
    tokenized_repliqa = tokenized_repliqa.train_test_split(test_size=0.1, seed=0)

    # Do the training.
    training_args = TrainingArguments(
        output_dir=f"models/{model_name.replace('/', '-')}-repliqa",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        num_train_epochs=5,
        weight_decay=0.01,
        push_to_hub=False,
    )

    data_collator = DefaultDataCollator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_repliqa["train"],
        eval_dataset=tokenized_repliqa["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()  # resume_from_checkpoint=True)
    trainer.save_model(f"models/{model_name.replace('/', '-')}-repliqa")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    args = docopt(__doc__)
    model_name = args["--model"] or "deepset/deberta-v3-base-squad2"
    print(
        f"Fine-tuning model: {model_name}. Will save to models/{model_name.replace('/', '-')}-repliqa"
    )
    run(model_name)
