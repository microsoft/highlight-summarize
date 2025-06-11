import judges
from textwrap import dedent
from typing import Any
from pydantic import BaseModel

from .utils import NOANSWER_PRED, query_llm, openai_client


class LLMJudgeResponse(BaseModel):
    rating: int | bool
    explanation: str

class LLMJudgeStructured():
    def __init__(self, model_name = "gpt-4.1-mini", temperature = 0, scale_min = 1, scale_max = 10):
        self.judge_name = f"LLMJudgeStructured-{model_name}"
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
        self.factors = ["correctness", "faithfulness", "naturalness"]
        # This can be overridden by the judge.
        self.correct = scale_max
        self.incorrect = scale_min

    def _format_response(self, responses: dict[LLMJudgeResponse]) -> dict[str, Any]:
        """Formats the response from the judge into a dictionary.
        """
        formatted_response = {}
        for factor, response in responses.items():
            formatted_response[f"{self.judge_name}_{factor}"] = response.rating
            formatted_response[f"{self.judge_name}_{factor}_explanation"] = response.explanation
        return formatted_response

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """A wrapper for the judge that takes an example and returns a formatted response.
        """
        assert "question" in example, "Example must contain 'question' field."
        assert "answer" in example, "Example must contain 'answer' field."
        assert "answer_pred" in example, "Example must contain 'answer_pred' field."

        # Check if unanswerable.
        unanswerable = (example["answer"] == NOANSWER_PRED)
        unanswerable_correct = (example["answer_pred"] == NOANSWER_PRED)
        if unanswerable:
            response = LLMJudgeResponse(
                rating=self.correct if unanswerable_correct else self.incorrect,
                explanation=NOANSWER_PRED
            )
            return self._format_response({factor: response for factor in self.factors})

        # Call judge.
        judgement = self._call_judge(
            input=example["question"],
            output=example["answer"],
            expected=example["answer_pred"]
        )

        return self._format_response(judgement)

    def _call_judge(self, input, output, expected) -> dict[LLMJudgeResponse]:
        """Provides a response for each of the factors specified in the prompt.
        """
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

####################################################################################
# Support for https://github.com/quotient-ai/judges.
# NOTE: the following hack is necessary to allow the library to support Azure OpenAI.
def patched_get_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    response_format: dict = None,
    response_model = None,
):
    """Monkey-patch the get_completion method to use the openai_client."""
    if response_model is not None:
        raise ValueError("response_model is not supported in this context. Use response_format instead.")

    return openai_client().beta.chat.completions.parse(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        response_format=response_format,
    )
judges.base.get_completion = patched_get_completion
# End of the hack.
####################################################################################

class PollMultihopCorrectness(LLMJudgeStructured):
    def __init__(self, model_name = "gpt-4.1-mini", temperature = 0, scale_min = 1, scale_max = 5):
        super().__init__(model_name=model_name, temperature=temperature, scale_min=scale_min, scale_max=scale_max)
        self.judge_name = f"PollMultihopCorrectness-{model_name}"
        self.correct = True
        self.incorrect = False
        self.judge = judges.PollMultihopCorrectness(model=model_name)

    def _call_judge(self, input, output, expected) -> dict[LLMJudgeResponse]:
        """Calls the PollMultihopCorrectness judge and formats the response.
        """
        # Call judge.
        model_response = self.judge.judge(
            input=input,
            output=output,
            expected=expected
        )

        # Format response.
        return {
            "correctness": LLMJudgeResponse(
                rating=bool(model_response.score),
                explanation=model_response.reasoning
            )
        }

class MTBenchChatBotResponseQuality(LLMJudgeStructured):
    def __init__(self, model_name = "gpt-4.1-mini", temperature = 0, scale_min = 1, scale_max = 10):
        super().__init__(model_name=model_name, temperature=temperature, scale_min=scale_min, scale_max=scale_max)
        self.judge_name = f"MTBenchChatBotResponseQuality-{model_name}"
        self.correct = scale_max
        self.incorrect = scale_min
        self.judge = judges.MTBenchChatBotResponseQuality(model=model_name)

    def _call_judge(self, input, output, expected) -> dict[LLMJudgeResponse]:
        """Calls the MTBenchChatBotResponseQuality judge and formats the response.
        """
        # Call judge.
        model_response = self.judge.judge(
            input=input,
            output=output,
            expected=expected
        )

        # Format response.
        return {
            "correctness": LLMJudgeResponse(
                rating=int(model_response.score),
                explanation=model_response.reasoning
            )
        }

class ReliableCIRelevance(MTBenchChatBotResponseQuality):
    def __init__(self, model_name = "gpt-4.1-mini", temperature = 0, scale_min = 1, scale_max = 10):
        super().__init__(model_name=model_name, temperature=temperature, scale_min=scale_min, scale_max=scale_max)
        self.judge_name = f"ReliableCIRelevance-{model_name}"
        self.correct = scale_max
        self.incorrect = scale_min
        self.judge = judges.ReliableCIRelevance(model=model_name)


JUDGES_MAP = {
    "LLMJudgeStructured": LLMJudgeStructured,
    "PollMultihopCorrectness": PollMultihopCorrectness,
    "MTBenchChatBotResponseQuality": MTBenchChatBotResponseQuality,
    "ReliableCIRelevance": ReliableCIRelevance,
}