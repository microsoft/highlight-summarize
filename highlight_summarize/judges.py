import judges as judges_lib
from typing import Any
from pydantic import BaseModel

from .utils import NOANSWER_PRED, query_llm, openai_client


class LLMJudgeResponse(BaseModel):
    rating: int | bool
    explanation: str


class LLMJudge:
    """A base class for LLM as judges."""

    def __init__(
        self,
        judge_name,
        correct=10,
        incorrect=1,
        factors=[],
    ):
        """Initialize the judge with a name, correctness ratings, and factors.

        Args:
            judge_name (str): The name of the judge.
            correct (int | bool): The rating for correct responses.
            incorrect (int | bool): The rating for incorrect responses.
            factors (list[str]): The factors to judge on.
        """
        self.judge_name = judge_name
        self.correct = correct
        self.incorrect = incorrect
        self.factors = factors

    def _format_response(
        self, responses: dict[str, LLMJudgeResponse]
    ) -> dict[str, Any]:
        """Formats the response from the judge into a dictionary."""
        formatted_response = {}
        for factor, response in responses.items():
            formatted_response[f"{self.judge_name}_{factor}_rating"] = response.rating
            formatted_response[f"{self.judge_name}_{factor}_explanation"] = (
                response.explanation
            )
        return formatted_response

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """A wrapper for the judge that takes an example and returns a formatted response."""
        assert "question" in example, "Example must contain 'question' field."
        assert "answer" in example, "Example must contain 'answer' field."
        assert "answer_pred" in example, "Example must contain 'answer_pred' field."

        # Check if unanswerable.
        unanswerable = example["answer"] == NOANSWER_PRED
        unanswerable_correct = example["answer_pred"] == NOANSWER_PRED
        if unanswerable:
            response = LLMJudgeResponse(
                rating=self.correct if unanswerable_correct else self.incorrect,
                explanation=NOANSWER_PRED,
            )
            return self._format_response({factor: response for factor in self.factors})

        # Call judge.
        judgement = self.call_judge(
            input=example["question"],
            output=example["answer"],
            expected=example["answer_pred"],
        )

        return self._format_response(judgement)

    def call_judge(self, input, output, expected) -> dict[str, LLMJudgeResponse]:
        """Provides a response for each of the factors specified in the prompt.
        This must be implemented by subclasses.
        """
        raise NotImplementedError()


####################################################################################
# Monkey-patching support for https://github.com/quotient-ai/judges.
# NOTE: the following hack is used to support Azure OpenAI.
def patched_get_completion(
    messages: list[dict[str, str]],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    response_format: dict | None = None,
    response_model=None,
):
    """Monkey-patch the get_completion method to use the openai_client."""
    if response_model is not None:
        raise ValueError(
            "response_model is not supported in this context. Use response_format instead."
        )

    return openai_client().beta.chat.completions.parse(
        model=model,
        messages=messages,  # type: ignore
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        response_format=response_format,  # type: ignore
    )


judges_lib.base.get_completion = patched_get_completion
judges_lib._client.get_completion = patched_get_completion
# End of monkey patching.
####################################################################################


class PollMultihopCorrectnessWrapper(LLMJudge):
    def __init__(self, model_name="gpt-4.1-mini"):
        super().__init__(
            judge_name=f"PollMultihopCorrectness-{model_name}",
            correct=True,
            incorrect=False,
            factors=["correctness"],
        )
        self.judge = judges_lib.PollMultihopCorrectness(model=model_name)

    def call_judge(self, input, output, expected) -> dict[str, LLMJudgeResponse]:
        """Calls the PollMultihopCorrectness judge and formats the response."""
        # Call judge.
        model_response = self.judge.judge(input=input, output=output, expected=expected)

        # Format response.
        return {
            "correctness": LLMJudgeResponse(
                rating=bool(model_response.score), explanation=model_response.reasoning
            )
        }


class MTBenchChatBotResponseQualityWrapper(LLMJudge):
    def __init__(self, model_name="gpt-4.1-mini"):
        super().__init__(
            judge_name=f"MTBenchChatBotResponseQuality-{model_name}",
            correct=10,
            incorrect=1,
            factors=["quality"],
        )
        self.judge = judges_lib.MTBenchChatBotResponseQuality(model=model_name)

    def call_judge(self, input, output, expected) -> dict[str, LLMJudgeResponse]:
        """Calls the MTBenchChatBotResponseQuality judge and formats the response."""
        # Call judge.
        model_response = self.judge.judge(input=input, output=output, expected=expected)

        # Format response.
        return {
            "quality": LLMJudgeResponse(
                rating=int(model_response.score), explanation=model_response.reasoning
            )
        }


class ReliableCIRelevanceWrapper(MTBenchChatBotResponseQualityWrapper):
    def __init__(self, model_name="gpt-4.1-mini"):
        LLMJudge.__init__(
            self,
            judge_name=f"ReliableCIRelevance-{model_name}",
            correct=3,
            incorrect=0,
            factors=["relevance"],
        )
        self.judge = judges_lib.ReliableCIRelevance(model=model_name)


JUDGES_MAP = {
    "PollMultihopCorrectness": PollMultihopCorrectnessWrapper,
    "MTBenchChatBotResponseQuality": MTBenchChatBotResponseQualityWrapper,
    "ReliableCIRelevance": ReliableCIRelevanceWrapper,
}
