from rapidfuzz import fuzz
from textwrap import dedent
from types import FunctionType
from pydantic import BaseModel
from transformers import pipeline
from text_chunker import sentences

from .qa import QAEvaluator, QAPrediction
from .utils import NOANSWER_PRED, FAILED_PRED

BASELINE_EXTRACTOR_PROMPT = dedent(
    "You are an expert research assistant."
    "You are given a context text and a question about it. Your task is to extract information from the context "
    "text that answers the question. If there is no information in the context that answers the question, "
    f"you must output a special token that indicates that the question is unanswerable: '{NOANSWER_PRED}'.\n"
    "If the answer is contained in parts of the text, you will output the relevant text extract(s), by prefixing each "
    "extract with a bullet point '-', and nothing else."
    "Context:\n"
    "{context}\n"
    "Question: {question_str}?\n"
)
BASELINE_SUMMARIZER_PROMPT = dedent(
    """You are given highlighted text from a document; the text extract is relevant to some
    question which you don't know. Figure out what question the text extract is trying
    to answer, and summarize the text extract in a concise manner in the form of an answer.
    Text extract:
    {text_extract}
    """
)

class HighlighterOutput(BaseModel):
    highlighter_extracted: str | None = None
    highlighter_llm_response: str | None = None
    highlighter_text_extracts: list[str] | None = None
    highlighter_fuzzmatch_scores: list[float] | None = None
    highlighter_score: float | None = None

class SummarizerOutput(BaseModel):
    answer_pred: str | None = None
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
        extractor_prompt: str = BASELINE_EXTRACTOR_PROMPT,
        summarizer_prompt: str = BASELINE_SUMMARIZER_PROMPT,
    ) -> None:
        super().__init__(
            model_name=model_name,
            openai_client=openai_client,
            temperature=temperature,
            n_trials=n_trials,
            sleep_time_between_retrials=sleep_time_between_retrials,
            max_sleep_time_between_retrials=max_sleep_time_between_retrials,
        )
        
        self.extractor_prompt = extractor_prompt
        self.summarizer_prompt = summarizer_prompt

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
        # Nothing to highlight.
        if NOANSWER_PRED in content:
            return HighlighterOutput(highlighter_llm_response=content)
        # Malformed output.
        if not content.strip().startswith("- "):
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
            summarizer_llm_response=str(raw_response),
            summarizer_llm_guessed_question=raw_response.guessed_question if hasattr(raw_response, "guessed_question") else None,
        )

class HSStructuredHighlighter(HSBaseline):
    """Highlighter that uses structured output.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.extractor_prompt = dedent(
            "I'd like for you to answer questions about a context text that will be provided."
            "I'll give you a pair with the form:\nContext: 'context text'\nQuestion: 'a question about the context'.\n"
            "First, tell me about your knowledge of the context and what information it contains, "
            "then, create an analysis of the context strictly using information contained in the text provided. "
            "If there is no information in the context that answers the question, "
            f"your answer _must_ be exactly '{NOANSWER_PRED}'.\n"
            "If the question can be answered, you must return the `answer` together with "
            "a list of text extracts (`text_extracts`) that allowed you to answer the question."
            "Here's the context and question for you to reason about and answer:\n"
            "Context:\n"
            "{context}\n"
            "Question: {question_str}?\n"
        )
 
    def call_highlighter(self, context_str: str, question_str: str) -> HighlighterOutput:
        """This highlither uses structured output to extract text from the context."""
        class LLMOutput(BaseModel):
            answer: str
            text_extracts: list[str]

        try:
            model_response = self.openai_client().beta.chat.completions.parse(
                messages=[
                    {"role": "user", "content": self.extractor_prompt.format(
                        context=context_str,
                        question_str=question_str,
                    )}
                ],
                temperature=self.temperature,
                model=self.model_name,
                response_format=LLMOutput,
            ).choices[0].message.parsed
        except Exception as e:
            print(f"Error in call_highlighter: {e}")
            return HighlighterOutput(
                highlighter_llm_response=f"Error: {e}"
            )

        # Nothing to highlight.
        if not model_response.answer or NOANSWER_PRED in model_response.answer:
            return HighlighterOutput(highlighter_llm_response=str(model_response))
        
        if not model_response.text_extracts:
            return HighlighterOutput(highlighter_llm_response=str(model_response))
        
        # Check if the text extracts are in the context.
        valid_text_extracts = []
        scores = []
        for text_extract in model_response.text_extracts:
            scores.append(fuzz.partial_ratio(text_extract, context_str))
            if scores[-1] >= 95:
                valid_text_extracts.append(text_extract)
        
        valid_text = "\n".join(valid_text_extracts)

        return HighlighterOutput(
            highlighter_extracted=valid_text.strip(),
            highlighter_llm_response=model_response.answer,
            highlighter_text_extracts=model_response.text_extracts,
            highlighter_fuzzmatch_scores=scores,
        )

class HSBERTExtractor(HSBaseline):
    """Highlighter that uses a BERT-based extractor.
    """
    def __init__(self, highlighter_threshold=0.5, highlighter_model='distilbert-base-cased-distilled-squad', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.extractor = pipeline("question-answering", model=highlighter_model)
        self.highlighter_threshold = highlighter_threshold
        # In case this hasn't been downloaded yet.
        import nltk
        nltk.download('punkt_tab')

    def call_highlighter(self, context_str: str, question_str: str) -> HighlighterOutput:
        """This highlighter uses a BERT-based extractor to extract text from the context."""
        try:
            model_response = self.extractor(
                question=question_str,
                context=context_str,
            )
        except Exception as e:
            print(f"Error in call_highlighter: {e}")
            return HighlighterOutput(
                highlighter_llm_response=f"Error: {e}"
            )

        # Nothing to highlight.
        if model_response['score'] < self.highlighter_threshold:
            return HighlighterOutput(
                highlighter_extracted=None,
                highlighter_score=model_response['score'],
                highlighter_llm_response=model_response['answer'],
            )

        # We return the full sentence that contains the answer.
        context = context_str.replace("\n", " ")
        for s in sentences(context):
            if s in context[model_response["start"]:]:
                extracted_sentence = s
                break
        else:
            raise ValueError("No sentence found containing the answer. This looks like a logical error in the code.")

        return HighlighterOutput(
            highlighter_extracted=extracted_sentence.strip(),
            highlighter_score=model_response['score'],
            highlighter_llm_response=model_response['answer'],
        )