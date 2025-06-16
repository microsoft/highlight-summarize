import random
from textwrap import dedent
from pydantic import BaseModel

from .utils import query_llm

class JudgeResponse(BaseModel):
    response_id: int | None
    explanation: str

class ComparisonJudge:
    def __init__(
        self, model_name="gpt-4.1-mini", temperature=0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt = dedent(
            """Please act as an impartial judge and evaluate the quality of two responses provided
            to the user question displayed below. Your evaluation should consider factors
            such as the correctness, faithfulness, and the naturalness of
            the response. Begin your evaluation by providing a short explanation. Be as objective as
            possible. After providing your explanation, select the response that you think is better.
            [Question]
            {input}
            [Response 1]
            {output_1}
            [Response 2]
            {output_2}
            [The End of Responses]
            """
        )

    def __call__(self, question: str, output_1: str, output_2: str) -> JudgeResponse:
        """Evaluate two responses to a question."""
        # Invert the answers at random.
        if random.random() > 0.5:
            inverted = False
            formatted_prompt = self.prompt.format(input=question, output_1=output_1, output_2=output_2)
        else:
            formatted_prompt = self.prompt.format(input=question, output_1=output_2, output_2=output_1)
            inverted = True

        model_response = query_llm(
            messages=[
                {"role": "system", "content": "You are an impartial judge."},
                {"role": "user", "content": formatted_prompt},
            ],
            temperature=self.temperature,
            model_name=self.model_name,
            response_format=JudgeResponse,
        )

        if not model_response.response_id in [1, 2]:
            return JudgeResponse(
                response_id=None,
                explanation="Error: Invalid response ID. Expected 1 or 2, got {}. Explanation: {}".format(
                    model_response.response_id, model_response.explanation),
            )

        if inverted:
            model_response.response_id = 1 if model_response.response_id == 2 else 2

        return model_response