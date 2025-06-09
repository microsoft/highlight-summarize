"""Standard Question Answering, as one would usually find in a RAG pipeline.
"""
import time
import json
import requests
from typing import Any
from pydantic import BaseModel

from .utils import NOANSWER_PRED, FAILED_PRED


class QAPrediction(BaseModel):
    """The prediction returned from a Q&A pipeline.
    """
    answer_pred: str
    # Model metadata.
    model_name: str | None = None
    temperature: float | None = None
    # Prediction metadata (can be augmented).
    llm_response: str | None = None

class QAEvaluator:
    def __init__(
        self,
        model_name: str | None,
        openai_client,
        temperature: float = 0.2,
        n_trials: int = 1,
        sleep_time_between_retrials: float = 1.0,
        max_sleep_time_between_retrials: float = 600.0,
    ) -> None:

        self.model_name = model_name
        self._base_system_prompt = (
            "You are an expert research assistant, skilled in answering questions "
            "concisely and precisely, using information provided by the user. "
        )
        self._base_user_prompt = (
            "I'd like for you to answer questions about a context text that will be provided."
            "I'll give you a pair with the form:\nContext: 'context text'\nQuestion: 'a question about the context'.\n"
            "First, tell me about your knowledge of the context and what information it contains, "
            "then, create an analysis of the context strictly using information contained in the text provided. "
            "Your knowledge about the context and the analysis must not be output. "
            "Finally, generate an explicit answer to the question that will be output. "
            "Make sure that the answer is the only output you provide, and the analysis of the context should be kept to yourself. "
            "Answer directly and do not prefix the answer with anything such as 'Answer:' nor 'The answer is:'. "
            "The answer has to be the only output you explicitly provide. "
            "The answer has to be as short, direct, and concise as possible. "
            f"If the answer to the question can not be obtained from the provided context paragraph, output '{NOANSWER_PRED}'. "
            "Here's the context and question for you to reason about and answer:\n"
            "Context:\n"
            "{context}\n"
            "Question: {question_str}?\n"
        )

        self.n_trials = n_trials
        self.sleep_time_between_retrials = sleep_time_between_retrials
        self.max_sleep_time_between_retrials = max_sleep_time_between_retrials
        self.openai_client = openai_client
        self.model_name = model_name
        self.temperature = temperature

    def call_model(self, context_str: str, question_str: str) -> QAPrediction:
        """Uses the LLM to answer the question based on the context provided.
        
        Args:
            context_str (str): The context text to use for answering the question.
            question_str (str): The question to be answered based on the context.
        Returns:
            tuple[str, dict[str, Any]]: A tuple containing the answer as well as any other metadata.
        """
        system_prompt_str = self._base_system_prompt
        user_prompt_str = self._base_user_prompt.format(
                context=context_str,
                question_str=question_str,
        )
        model_response = self.openai_client().chat.completions.create(
                    messages=[
                                {
                                    "role": "system",
                                    "content": system_prompt_str,
                                },
                                {"role": "user", "content": user_prompt_str},
                            ],
                    temperature=self.temperature,
                    model=self.model_name,
                )

        llm_response = model_response.choices[0].message.content
        if not llm_response:
            return QAPrediction(answer_pred=FAILED_PRED, llm_response=llm_response)

        return QAPrediction(
            answer_pred=llm_response.split(":")[-1].strip(),
            llm_response=llm_response,
            model_name=self.model_name,
            temperature=self.temperature,
        )

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        if "document_extracted" in example:
            context_str = example["document_extracted"]
        elif "entity_pages" in example:
            context_str = ("\n\n").join(example["entity_pages"]["wiki_context"])
        else:
            raise ValueError("Unknown data format. Can't read 'document_extracted' or 'entity_pages' fields.")
        question_str = example["question"]
        
        for trial in range(self.n_trials):
            try:
                prediction = self.call_model(context_str, question_str)
                break
            except (
                KeyError,
                IndexError,
                json.JSONDecodeError,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                print(f"Trial: {trial}: {e}")
                prediction = QAPrediction(answer_pred="")
                sleep_time = min(
                    self.max_sleep_time_between_retrials,
                    self.sleep_time_between_retrials * (2 ** (trial + 1)),
                )
                time.sleep(sleep_time)

        return prediction.model_dump()