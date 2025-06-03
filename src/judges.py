import json
from textwrap import dedent
from typing import Dict, Any
from pydantic import BaseModel
from types import FunctionType

from .utils import NOANSWER_PRED

JUDGES = {
    "MT-bench-QA": dedent(
            """
            [System]
            Please act as an impartial judge and evaluate the quality of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should consider factors
            such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
            the response. Begin your evaluation by providing a short explanation. Be as objective as
            possible. After providing your explanation, please rate the response on a scale of {scale_min} to {scale_max}.
            [Question]
            {input}
            [Expected Answer]
            {expected}
            [The Start of Assistant's Answer]
            {output}
            [The End of Assistant's Answer]
            """
        ),
    "Correctness": dedent(
            """
            [System]
            Please act as an impartial judge and evaluate the correctness of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should only consider whether 
            the correct information is contained in the answer. Begin your evaluation by providing a short explanation. Be as objective as
            possible. After providing your explanation, please rate the response on a scale of {scale_min} to {scale_max}.
            [Question]
            {input}
            [Expected Answer]
            {expected}
            [The Start of Assistant's Answer]
            {output}
            [The End of Assistant's Answer]
            """
        ),
    # NOTE: this judge does not use the expected answer.
    "Naturalness": dedent(
            """
            [System]
            Please act as an impartial judge and evaluate the naturalness of the response provided by an
            AI assistant to the user question displayed below.
            Your evaluation should consider whether the provided answer sounds like an appropriate response
            to the question, and whether it is written in a natural way. Disregard whether the answer is correct or not.
            Begin your evaluation by providing a short explanation. Be as objective as
            possible. After providing your explanation, please rate the response on a scale of {scale_min} to {scale_max}.
            [Question]
            {input}
            [The Start of Assistant's Answer]
            {output}
            [The End of Assistant's Answer]
            """
        ),
}

class LLMJudgeResponse(BaseModel):
    rating: int
    explanation: str

class LLMJudge:
    def __init__(self, judge_name: str, openai_client: FunctionType, model_name: str = "gpt-4.1-mini", temperature: float = 0, scale_min: int = 1, scale_max: int = 5) -> None:
        self.judge_name = judge_name
        self.judge_prompt = JUDGES[judge_name]
        self.model_name = model_name
        self.openai_client = openai_client
        self.temperature = temperature
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, example: dict[str, Any]) -> Dict[str, Any]:
        """
        A wrapper for the judge that takes an example and returns a formatted response.
        """
        assert "question" in example, "Input must contain 'question' field."
        assert "answer" in example, "Output must contain 'answer' field."
        assert "answer_pred" in example, "Output must contain 'answer_pred' field. You must run QAEvaluator first."

        judgement = self._call_judge(
            input=example["question"],
            output=example["answer_pred"],
            expected=example["answer"]
        )

        if isinstance(judgement, dict):
            # We have more than one ratings.
            ratings = {}
            for factor, response in judgement.items():
                ratings[f"{self.judge_name}_{factor}"] = response.rating
                ratings[f"{self.judge_name}_{factor}_explanation"] = response.explanation
            return ratings

        # We have a single rating.
        return {
            f"{self.judge_name}_rating": judgement.rating,
            f"{self.judge_name}_explanation": judgement.explanation,
        }

    def _call_judge(self, input: str, output: str, expected: str) -> LLMJudgeResponse:
        """
        Calls the judge with the provided input, output, and expected answer.
        """
        # Check if unanswerable.
        unanswerable = (expected == NOANSWER_PRED)
        unanswerable_correct = (output == NOANSWER_PRED)
        if unanswerable:
            return LLMJudgeResponse(
                rating=self.scale_max if unanswerable_correct else self.scale_min,
                explanation=NOANSWER_PRED
            )

        # Call judge.
        class LLMRatingOutput(BaseModel):
            rating: int
            explanation: str

        judge_prompt = self.judge_prompt.format(
            input=input,
            expected=expected,
            output=output,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )
        model_response = self.openai_client().beta.chat.completions.parse(
            messages=[
                {
                    "role": "system",
                    "content": "You are an impartial judge evaluating the quality of AI responses.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            model=self.model_name,
            response_format=LLMRatingOutput,
            temperature=self.temperature,
        ).choices[0].message.parsed
        
        return LLMJudgeResponse(
            rating=model_response.rating if hasattr(model_response, 'rating') else None,
            explanation=model_response.explanation if hasattr(model_response, 'explanation') else model_response
        )

class LLMJudgeStructured(LLMJudge):
    def __init__(self, openai_client, model_name = "gpt-4.1-mini", temperature = 0, scale_min = 1, scale_max = 5):
        super().__init__("MT-bench-QA", openai_client, model_name, temperature, scale_min, scale_max)
        self.judge_prompt = dedent(
            """
            [System]
            Please act as an impartial judge and evaluate the quality of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should consider the following
            factors:
            Correctness: How much of the Expected Answer (ground truth) is contained in the assistant's answer?
            Faithfulness: How much of the assistant's answer matches the Expected Answer (ground truth)?
            Naturalness: How natural does the assistant's response sound if provided as a response to the Question?
            Begin your evaluation by providing a short explanation. Be as objective as possible.
            After providing your explanation, please rate the response on a scale of {scale_min} to {scale_max}
            for each of the factors mentioned above.
            [Question]
            {input}
            [Expected Answer]
            {expected}
            [The Start of Assistant's Answer]
            {output}
            [The End of Assistant's Answer]
            """
        )
        self.output_fields = [
            f"{self.judge_name}_{factor}_rating" for factor in ["correctness", "faithfulness", "naturalness"]
        ] + [
            f"{self.judge_name}_{factor}_explanation" for factor in ["correctness", "faithfulness", "naturalness"]
        ]

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

        judge_prompt = self.judge_prompt.format(
            input=input,
            expected=expected,
            output=output,
            scale_min=self.scale_min,
            scale_max=self.scale_max,
        )

        model_response = self.openai_client().beta.chat.completions.parse(
            messages=[
                {
                    "role": "system",
                    "content": "You are an impartial judge evaluating the quality of AI responses.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            model=self.model_name,
            response_format=LLMRatingOutput,
            temperature=self.temperature,
        ).choices[0].message.parsed

        return {
            factor: LLMJudgeResponse(
                rating=getattr(model_response, factor) if hasattr(model_response, factor) else None,
                explanation=getattr(model_response, f"{factor}_explanation") if hasattr(model_response, f"{factor}_explanation") else model_response
            )
            for factor in ["correctness", "faithfulness", "naturalness"]
        }