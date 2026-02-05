import json
import random
from enum import Enum
from textwrap import dedent
from typing import Any
from pydantic import BaseModel

from .utils import query_llm, create_batch_request


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
        self,
        model_name="gpt-4.1-mini",
        temperature=0,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.prompt = dedent(
            """Please act as an impartial judge and evaluate the quality of two responses provided
            to the user question displayed below. Your evaluation must consider
            the factual correctness of the response to the question and the relevance of the response
            as an answer to the question.
            Do NOT prefer longer or more detailed answers: relevance and correctness are the only criteria.
            Begin your evaluation by providing a short explanation. Be as objective as
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

    def _format_messages(
        self, question: str, output_1: str, output_2: str
    ) -> list[dict[str, str]]:
        """Format the messages for the judge."""
        formatted_prompt = self.prompt.format(
            input=question, output_1=output_1, output_2=output_2
        )
        return [
            {"role": "system", "content": "You are an impartial judge."},
            {"role": "user", "content": formatted_prompt},
        ]

    def create_batch_request(
        self,
        custom_id: str,
        question: str,
        output_1: str,
        output_2: str,
    ) -> tuple[dict[str, Any], bool]:
        """
        Create a batch request for comparing two outputs.

        Returns:
            Tuple of (batch_request_dict, inverted_flag)
            The inverted_flag indicates if output_1 and output_2 were swapped.
        """
        # Randomly invert to reduce position bias
        inverted = random.random() > 0.5
        if inverted:
            output_1, output_2 = output_2, output_1

        messages = self._format_messages(question, output_1, output_2)

        # Encode inversion flag in custom_id: "{original_id}:{0|1}"
        encoded_id = f"{custom_id}:{1 if inverted else 0}"

        request = create_batch_request(
            custom_id=encoded_id,
            messages=messages,
            model_name=self.model_name,
            temperature=self.temperature,
            response_format=JudgeResponse,
        )
        return request, inverted

    @staticmethod
    def parse_batch_response(result: dict[str, Any]) -> tuple[str, JudgeResponse]:
        """
        Parse a batch response and return the original custom_id and JudgeResponse.

        Automatically handles inversion reversal based on the encoded flag.

        Returns:
            Tuple of (original_custom_id, JudgeResponse)
        """
        encoded_id = result["custom_id"]
        original_id, inverted_str = encoded_id.rsplit(":", 1)
        inverted = inverted_str == "1"

        response_body = result["response_body"]
        content = response_body["choices"][0]["message"]["content"]
        parsed = JudgeResponse.model_validate(json.loads(content))

        # Reverse inversion if needed
        if inverted:
            if parsed.preference == ResponseChoice.response_1:
                parsed.preference = ResponseChoice.response_2
            elif parsed.preference == ResponseChoice.response_2:
                parsed.preference = ResponseChoice.response_1

        return original_id, parsed

    def __call__(self, question: str, output_1: str, output_2: str) -> JudgeResponse:
        """Evaluate two responses to a question."""
        # Invert the answers at random.
        if random.random() > 0.5:
            inverted = False
            formatted_prompt = self.prompt.format(
                input=question, output_1=output_1, output_2=output_2
            )
        else:
            formatted_prompt = self.prompt.format(
                input=question, output_1=output_2, output_2=output_1
            )
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

        if not isinstance(model_response, JudgeResponse):
            raise ValueError(f"Expected JudgeResponse, got {type(model_response)}")

        # If inverted, swap the responses.
        if inverted:
            if model_response.preference == ResponseChoice.response_1:
                model_response.preference = ResponseChoice.response_2
            elif model_response.preference == ResponseChoice.response_2:
                model_response.preference = ResponseChoice.response_1

        return model_response
