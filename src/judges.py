import json
from textwrap import dedent
from typing import Dict, Any
from pydantic import BaseModel
from types import FunctionType

JUDGES = {
    "MT-bench-QA": dedent(
            """
            [System]
            Please act as an impartial judge and evaluate the quality of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should consider factors
            such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
            the response. Begin your evaluation by providing a short explanation. Be as objective as
            possible. After providing your explanation, please rate the response on a scale of 1 to 10.
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
            AI assistant to the user question displayed below. Your evaluation should consider the correctness
            and the completeness with respect to the Expected Answer. Begin your evaluation by providing a short explanation. Be as objective as
            possible. After providing your explanation, please rate the response on a scale of 1 to 10.
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
            possible. After providing your explanation, please rate the response on a scale of 1 to 10.
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
    def __init__(self, judge_name: str, model_name: str, openai_client: FunctionType) -> None:
        self.judge_name = judge_name
        self.judge_prompt = JUDGES[judge_name]
        self.model_name = model_name
        self.openai_client = openai_client
    
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
        Evaluates the quality of an AI response to a question using all the judges.
        """
        assert "question" in example, "Input must contain 'question' field."
        assert "answer" in example, "Output must contain 'answer' field."
        assert "answer_pred" in example, "Output must contain 'answer_pred' field. You must run QAEvaluator first."
        return self._call_judge(
            input=example["question"],
            output=example["answer_pred"],
            expected=example["answer"]
        )

    def _call_judge(self, input: str, output: str, expected: str) -> Dict[str, Any]:
        """
        Calls the judge with the provided input, output, and expected answer.
        """
        # Check if unanswerable.
        unanswerable = (expected == "The answer is not found in the document.")
        unanswerable_correct = (output == "UNANSWERABLE")
        if unanswerable:
            return self._format_response(
                rating=10 if unanswerable_correct else 1,
                raw_response="Unanswerable question."
            )

        # Call judge.
        judge_prompt = self.judge_prompt.format(
            input=input,
            expected=expected,
            output=output,
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
        )
        raw_response = model_response.choices[0].message.content.strip()

        try:
            response_json = json.loads(raw_response)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {raw_response}")
        
        return self._format_response(
            rating=response_json.get("rating", 1),
            raw_response=response_json.get("raw_response", raw_response)
        )