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
            [The Start of Assistant’s Answer]
            {output}
            [The End of Assistant’s Answer]
            """
        )
}

class QAJudgeResponse(BaseModel):
    rating: int
    raw_response: str

class QAJudge:
    def __init__(self, model_name: str, openai_client: FunctionType, judges: list[str] | None = None) -> None:
        self.model_name = model_name
        self.openai_client = openai_client
        if judges is None:
            self.judges = JUDGES
        else:
            self.judges = {judge_name: JUDGES[judge_name] for judge_name in JUDGES}

    def __call__(self, example: dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the quality of an AI response to a question using all the judges.
        """
        assert "question" in example, "Input must contain 'question' field."
        assert "answer" in example, "Output must contain 'answer' field."
        assert "answer_pred" in example, "Output must contain 'answer_pred' field. You must run QAEvaluator first."

        for judge_name in self.judges:
            res = self._call_judge(
                judge_name=judge_name,
                input=example["question"],
                output=example["answer_pred"],
                expected=example["answer"]
            )
            example[f"{judge_name}_rating"] = res["rating"]
            example[f"{judge_name}_raw_response"] = res["raw_response"]

        return example
    
    def _call_judge(self, judge_name: str, input: str, output: str, expected: str) -> dict[str, Any]:
        # Check if unanswerable.
        unanswerable = (expected == "The answer is not found in the document.")
        unanswerable_correct = (output == "UNANSWERABLE")
        if unanswerable:
            return {
                "rating": 10 if unanswerable_correct else 1,
                "raw_response": "Unanswerable question."
            }

        # Call judge.
        user_prompt = self.judges[judge_name].format(
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
                {"role": "user", "content": user_prompt},
            ],
            model=self.model_name,
            response_format=QAJudgeResponse,
        )
        raw_response = model_response.choices[0].message.content.strip()
        # print(f"Raw response: {raw_response}")
        try:
            response_json = json.loads(raw_response)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from response: {raw_response}")
            return {"rating": 0, "raw_response": raw_response}
        return response_json