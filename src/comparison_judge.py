import random
from enum import Enum
from textwrap import dedent
from pydantic import BaseModel

from .utils import query_llm

class ResponseChoice(Enum):
    response_1 = "1"
    response_2 = "2"
    tie = "tie"
    neither = "neither"

class JudgeResponse(BaseModel):
    preference: ResponseChoice
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
            If you think both responses are equally good, select "tie". If you think both responses are equally bad,
            select "neither".
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

        # If inverted, swap the responses.
        if inverted:
            if model_response.preference == ResponseChoice.response_1:
                model_response.preference = ResponseChoice.response_2
            elif model_response.preference == ResponseChoice.response_2:
                model_response.preference = ResponseChoice.response_1

        return model_response