"""Standard Question Answering, as one would usually find in a RAG pipeline."""

from typing import Any
from pydantic import BaseModel

from .utils import NOANSWER_PRED, FAILED_PRED, query_llm


class QAPrediction(BaseModel):
    """The prediction returned from a Q&A pipeline."""

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
        temperature: float = 0.2,
        sleep_time_between_retrials: float = 1.0,
        max_sleep_time_between_retrials: float = 600.0,
    ) -> None:
        self._base_system_prompt = (
            "You are an expert research assistant, skilled in answering questions "
            "concisely and precisely, using information provided by the user. "
        )
        self._base_user_prompt = (
            "Answer the following question based on the provided context.\n"
            "The answer must be 1-2 sentences, as short, direct, and concise as possible.\n"
            "Do not prefix the answer with anything such as 'Answer:' or 'The answer is:'.\n"
            f"If the answer cannot be obtained from the context, output '{NOANSWER_PRED}'.\n"
            "Context:\n"
            "{context}\n"
            "Question: {question_str}?\n"
        )

        self.sleep_time_between_retrials = sleep_time_between_retrials
        self.max_sleep_time_between_retrials = max_sleep_time_between_retrials
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
        assert self.model_name is not None, "model_name cannot be None"

        system_prompt_str = self._base_system_prompt
        user_prompt_str = self._base_user_prompt.format(
            context=context_str,
            question_str=question_str,
        )

        model_response = query_llm(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt_str,
                },
                {"role": "user", "content": user_prompt_str},
            ],
            temperature=self.temperature,
            model_name=self.model_name,
        )

        if not model_response:
            return QAPrediction(answer_pred=FAILED_PRED, llm_response=model_response)

        return QAPrediction(
            answer_pred=model_response.strip(),
            llm_response=model_response,
            model_name=self.model_name,
            temperature=self.temperature,
        )

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        if "document_extracted" in example:
            context_str = example["document_extracted"]
        elif "entity_pages" in example:
            context_str = ("\n\n").join(example["entity_pages"]["wiki_context"])
        else:
            raise ValueError(
                "Unknown data format. Can't read 'document_extracted' or 'entity_pages' fields."
            )
        question_str = example["question"]

        try:
            prediction = self.call_model(context_str, question_str)
        except Exception as e:
            prediction = QAPrediction(answer_pred=FAILED_PRED)

        return prediction.model_dump()
