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
    raw_response: str

class LLMJudge:
    def __init__(self, judge_name: str, openai_client: FunctionType, model_name: str = "gpt-4.1-mini", temperature: float = 0.1, scale_min: int = 1, scale_max: int = 5) -> None:
        self.judge_name = judge_name
        self.judge_prompt = JUDGES[judge_name]
        self.model_name = model_name
        self.openai_client = openai_client
        self.temperature = temperature
        self.scale_min = scale_min
        self.scale_max = scale_max
    
    def _format_response(self, rating: int, raw_response: str) -> Dict[str, Any]:
        """
        Formats the response from the judge into a dictionary.
        """
        return {
            f"{self.judge_name}-rating": rating,
            f"{self.judge_name}-raw_response": raw_response
        }

    def __call__(self, example: dict[str, Any]) -> Dict[str, Any]:
        """
        A wrapper for the judge that takes an example and returns a formatted response.
        """
        assert "question" in example, "Input must contain 'question' field."
        assert "answer" in example, "Output must contain 'answer' field."
        assert "answer_pred" in example, "Output must contain 'answer_pred' field. You must run QAEvaluator first."
        rating, raw_response = self._call_judge(
            input=example["question"],
            output=example["answer_pred"],
            expected=example["answer"]
        )
        return self._format_response(rating=rating, raw_response=raw_response)

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
                raw_response=NOANSWER_PRED
            )

        # Call judge.
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
            response_format=LLMJudgeResponse,
            temperature=self.temperature,
        )
        raw_response = model_response.choices[0].message.content.strip()

        try:
            response_json = json.loads(raw_response)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {raw_response}")

        return LLMJudgeResponse(
            rating=response_json.get("rating", None),
            raw_response=raw_response
        )