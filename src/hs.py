from rapidfuzz import fuzz
from textwrap import dedent
from types import FunctionType
from pydantic import BaseModel

from .qa import QAEvaluator, QAPrediction
from .utils import NOANSWER_PRED, FAILED_PRED

class HighlighterOutput(BaseModel):
    highlighter_extracted: str | None = None
    highlighter_llm_response: str | None = None
    highlighter_text_extracts: list[str] | None = None
    highlighter_fuzzmatch_scores: list[float] | None = None

class SummarizerOutput(BaseModel):
    anser_pred: str | None = None
    summarizer_llm_response: str | None = None
    summarizer_llm_guessed_question: str | None = None

class HSBaselinePrediction(QAPrediction, HighlighterOutput, SummarizerOutput):
    """Prediction made by the H&S pipeline.
    """
    pass


class HSBaseline(QAEvaluator):
    """H&S based on two LLMs.
    """
    def __init__(
        self,
        model_name: str,
        openai_client: FunctionType,
        temperature: float = 0.3,
        n_trials: int = 1,
        sleep_time_between_retrials: float = 1.0,
        max_sleep_time_between_retrials: float = 600.0,
    ) -> None:
        super().__init__(
            model_name=model_name,
            openai_client=openai_client,
            temperature=temperature,
            n_trials=n_trials,
            sleep_time_between_retrials=sleep_time_between_retrials,
            max_sleep_time_between_retrials=max_sleep_time_between_retrials,
        )
        
        self.extractor_prompt = dedent(
            "You are an expert research assistant."
            "You are tasked with highlighting relevant text extract(s) from a context text."
            "Your output must be (verbatim) a list of the text extract(s) that answer the question."
            "You will output the text extract(s), by prefixing each extract with a bullet point '-',"
            "and nothing else."
            f"Important: if no part of the text answers the question, you must output '{NOANSWER_PRED}'.\n"
            "Context:\n"
            "{context}\n"
            "Question: {question_str}?\n"
            )
        self.summarizer_prompt = dedent(
            """You are given highlighted text from a document; the text extract is relevant to some
            question which you don't know. Figure out what question the text extract is trying
            to answer, and summarize the text extract in a concise manner in the form of an answer.
            Text extract:
            {text_extract}
            """)

    def call_model(self, context_str: str, question_str: str) -> HSBaselinePrediction:
        highlighted = self.call_highlighter(context_str, question_str)

        if not highlighted.highlighter_extracted:
            return HSBaselinePrediction(
                # Refuse to answer (no highlight).
                answer_pred=NOANSWER_PRED,
                **highlighted.model_dump(),
            )

        summarized = self.call_summarizer(highlighted)

        return HSBaselinePrediction(
            **highlighted.model_dump(),
            **summarized.model_dump(),
            model_name=self.model_name,
            temperature=self.temperature,
        )
        
    def call_highlighter(self, context_str: str, question_str: str) -> HighlighterOutput:
        model_response = self.openai_client().chat.completions.create(
            messages=[
                {"role": "user", "content": self.extractor_prompt.format(
                    context=context_str,
                    question_str=question_str,
                )}
            ],
            temperature=self.temperature,
            model=self.model_name,
        )

        # No response.
        if not model_response.choices or not model_response.choices[0].message.content:
            return HighlighterOutput()
        content = model_response.choices[0].message.content
        # Malformed output.
        if not content.strip().startswith("- "):
            return HighlighterOutput(highlighter_llm_response=content)
        # Nothing to highlight.
        if NOANSWER_PRED in text_extracts:
            return HighlighterOutput(highlighter_llm_response=content)
        # Extract the text extract(s) from the response.
        text_extracts = content.strip().split("\n")
        text_extracts = [extract.strip().lstrip("- ") for extract in text_extracts]
        if not text_extracts:
            return HighlighterOutput(highlighter_llm_response=content)
        
        # Check if the text extracts are in the context.
        valid_text_extracts = []
        scores = []
        for text_extract in text_extracts:
            scores.append(fuzz.partial_ratio(text_extract, context_str))
            if scores[-1] >= 95:
                valid_text_extracts.append(text_extract)
        
        valid_text = "\n".join(valid_text_extracts)

        return HighlighterOutput(
            highlighter_extracted=valid_text.strip(),
            highlighter_llm_response=content,
            highlighter_text_extracts=text_extracts,
            highlighter_fuzzmatch_scores=scores,
        )

    def call_summarizer(self, text_extract: str) -> SummarizerOutput:
        # Structured output: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python-secure%2Cdotnet-entra-id&pivots=programming-language-python.
        class LLMSummarizerOutput(BaseModel):
            guessed_question: str
            answer: str
        raw_response = self.openai_client().beta.chat.completions.parse(
            messages=[
                {"role": "user", "content": self.summarizer_prompt.format(
                    text_extract=text_extract,
                )}
            ],
            temperature=self.temperature,
            model=self.model_name,
            response_format=LLMSummarizerOutput,
        ).choices[0].message.parsed

        return SummarizerOutput(
            # Failed prediction if the LLM gives no answer.
            answer_pred=raw_response.answer if hasattr(raw_response, "answer") else FAILED_PRED,
            summarizer_llm_response=raw_response,
            summarizer_llm_guessed_question=raw_response.guessed_question if hasattr(raw_response, "guessed_question") else None,
        )