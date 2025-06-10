from textwrap import dedent
from typing import Any
from pydantic import BaseModel

from .utils import NOANSWER_PRED, query_llm

class LLMJudgeResponse(BaseModel):
    rating: int
    explanation: str

class LLMJudgeStructured():
    def __init__(self, model_name = "gpt-4.1-mini", temperature = 0, scale_min = 1, scale_max = 5):
        self.judge_name = f"LLM-{model_name}"
        self.model_name = model_name
        self.temperature = temperature
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.user_prompt = dedent("""
            [System]
            Please act as an impartial judge and evaluate the quality of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should consider factors
            such as the correctness, faithfulness, and the naturalness of
            the response. Begin your evaluation by providing a short explanation. Be as objective as
            possible. After providing your explanation, please rate the response on a scale of {scale_min} to {scale_max}.
            [Question]
            {input}
            [Expected]
            {expected}
            [The Start of Assistant’s Answer]
            {output}
            [The End of Assistant’s Answer]""")
        self.output_fields = [
            f"{self.judge_name}_{factor}" for factor in ["correctness", "faithfulness", "naturalness"]
        ] + [
            f"{self.judge_name}_{factor}_explanation" for factor in ["correctness", "faithfulness", "naturalness"]
        ]

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """A wrapper for the judge that takes an example and returns a formatted response.
        """
        assert "question" in example, "Example must contain 'question' field."
        assert "answer" in example, "Example must contain 'answer' field."
        assert "answer_pred" in example, "Example must contain 'answer_pred' field."

        judgement = self._call_judge(
            input=example["question"],
            output=example["answer"],
            expected=example["answer_pred"]
        )

        ratings = {}
        for factor, response in judgement.items():
            ratings[f"{self.judge_name}_{factor}"] = response.rating
            ratings[f"{self.judge_name}_{factor}_explanation"] = response.explanation
        return ratings

    def _call_judge(self, input, output, expected) -> dict[LLMJudgeResponse]:
        """Provides a response for each of the factors specified in the prompt.
        """
        # Check if unanswerable.
        unanswerable = (expected == NOANSWER_PRED)
        unanswerable_correct = (output == NOANSWER_PRED)
        if unanswerable:
            response = LLMJudgeResponse(
                rating=self.scale_max if unanswerable_correct else self.scale_min,
                explanation=NOANSWER_PRED
            )
            return {factor: response for factor in ["correctness", "faithfulness", "naturalness"]}

        # Call judge.
        # NOTE: keeping a flat structure because I fear the LLM
        # may otherwise hallucinate.
        class LLMRatingOutput(BaseModel):
            correctness_explanation: str
            correctness: int
            faithfulness: int
            faithfulness_explanation: str
            naturalness: int
            naturalness_explanation: str

        user_prompt = self.user_prompt.format(
            input=input,
            expected=expected,
            output=output,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )

        model_response = query_llm(
            messages=[
                {"role": "system", "content": "You are an impartial judge."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.temperature,
            model_name=self.model_name,
            response_format=LLMRatingOutput
        )

        return {
            factor: LLMJudgeResponse(
                rating=getattr(model_response, factor) if hasattr(model_response, factor) else None,
                explanation=getattr(model_response, f"{factor}_explanation") if hasattr(model_response, f"{factor}_explanation") else model_response
            )
            for factor in ["correctness", "faithfulness", "naturalness"]
        }