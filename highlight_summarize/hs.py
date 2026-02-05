import torch
from rapidfuzz import fuzz
from textwrap import dedent
from pydantic import BaseModel
from text_chunker import sentences

from .qa import QAEvaluator, QAPrediction
from .utils import NOANSWER_PRED, FAILED_PRED, query_llm


class HighlighterOutput(BaseModel):
    highlighter_extracted: str | None = None
    highlighter_llm_response: str | None = None
    highlighter_text_extracts: list[str] | None = None
    highlighter_valid_text_extracts: list[str] | None = None
    highlighter_fuzzmatch_scores: list[float] | None = None
    highlighter_score: float | None = None


class SummarizerOutput(BaseModel):
    answer_pred: str
    summarizer_llm_response: str | None = None
    summarizer_llm_guessed_questions: list[str] | None = None


class HSBaselinePrediction(QAPrediction, HighlighterOutput, SummarizerOutput):
    """Prediction made by the H&S pipeline."""

    highlighter_model_name: str | None = None
    summarizer_model_name: str | None = None


def fuzzmatch_extract(
    text_extract: str, context_str: str, threshold: float = 95.0
) -> tuple[float, str | None]:
    """Fuzzy match a text extract against the context string.

    Returns the match score and the matched text if above threshold, else None.
    """
    alignment = fuzz.partial_ratio_alignment(text_extract, context_str)

    if not alignment:
        return 0.0, None

    if alignment.score >= threshold:
        return alignment.score, context_str[alignment.dest_start : alignment.dest_end]

    return alignment.score, None


class HSBaseline(QAEvaluator):
    """H&S based on two LLMs."""

    extractor_prompt: str = dedent(
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
    summarizer_system_prompt: str = (
        "You are an expert research assistant, skilled in answering questions "
        "concisely and precisely, using information provided by the user. "
    )
    summarizer_prompt: str = dedent(
        """You are given highlighted text from a document; the text extract is relevant to some
        question which you don't know. Follow these steps:
        1. Guess what question the text extract is trying to answer (you will provide 5-10 guesses).
        2. Then, summarize the text extract as an answer to the most likely guessed question.
        The answer must be 1-2 sentences, as short, direct, and concise as possible.
        Text extract:
        {text_extract}
        """
    )

    def __init__(
        self,
        highlighter_model_name: str,
        summarizer_model_name: str,
        temperature: float = 0.2,
        sleep_time_between_retrials: float = 1.0,
        max_sleep_time_between_retrials: float = 600.0,
        extractor_prompt: str | None = None,
        summarizer_prompt: str | None = None,
        min_highlighted_words: int | None = None,
    ) -> None:
        super().__init__(
            model_name=None,
            temperature=temperature,
            sleep_time_between_retrials=sleep_time_between_retrials,
            max_sleep_time_between_retrials=max_sleep_time_between_retrials,
        )

        self.extractor_prompt = extractor_prompt or self.extractor_prompt
        self.summarizer_prompt = summarizer_prompt or self.summarizer_prompt
        self.highlighter_model_name = highlighter_model_name
        self.summarizer_model_name = summarizer_model_name
        self.min_highlighted_words = min_highlighted_words

    def _validate_text_extract(self, text_extract: str) -> bool:
        """Check if a text extract has enough words."""
        if not self.min_highlighted_words:
            return True
        return len(text_extract.split()) >= self.min_highlighted_words

    def call_model(self, context_str: str, question_str: str) -> HSBaselinePrediction:
        highlighted = self.call_highlighter(context_str, question_str)

        if not highlighted.highlighter_extracted:
            return HSBaselinePrediction(
                # Refuse to answer (no highlight).
                answer_pred=NOANSWER_PRED,
                **highlighted.model_dump(),
            )

        summarized = self.call_summarizer(highlighted.highlighter_extracted)

        return HSBaselinePrediction(
            **highlighted.model_dump(),
            **summarized.model_dump(),
            highlighter_model_name=self.highlighter_model_name,
            summarizer_model_name=self.summarizer_model_name,
            temperature=self.temperature,
        )

    def call_highlighter(
        self, context_str: str, question_str: str
    ) -> HighlighterOutput:
        """This highlighter uses an LLM to extract text from the context."""
        model_response = query_llm(
            messages=[
                {
                    "role": "user",
                    "content": self.extractor_prompt.format(
                        context=context_str,
                        question_str=question_str,
                    ),
                }
            ],
            temperature=self.temperature,
            model_name=self.highlighter_model_name,
        )

        # No response.
        if not model_response:
            return HighlighterOutput()
        # Nothing to highlight.
        if NOANSWER_PRED in model_response:
            return HighlighterOutput(highlighter_llm_response=model_response)
        # Malformed output.
        if not model_response.strip().startswith("- "):
            return HighlighterOutput(highlighter_llm_response=model_response)
        # Extract the text extract(s) from the response.
        text_extracts = model_response.strip().split("\n")
        text_extracts = [extract.strip().lstrip("- ") for extract in text_extracts]
        if not text_extracts:
            return HighlighterOutput(highlighter_llm_response=model_response)

        # Check if the text extracts are in the context.
        valid_text_extracts = []
        scores = []
        for text_extract in text_extracts:
            score, matched_text = fuzzmatch_extract(text_extract, context_str)
            scores.append(score)
            if matched_text and self._validate_text_extract(matched_text):
                valid_text_extracts.append(matched_text)

        valid_text = "\n".join(valid_text_extracts)

        return HighlighterOutput(
            highlighter_extracted=valid_text.strip(),
            highlighter_llm_response=model_response,
            highlighter_text_extracts=text_extracts,
            highlighter_valid_text_extracts=valid_text_extracts,
            highlighter_fuzzmatch_scores=scores,
        )

    def call_summarizer(self, text_extract: str) -> SummarizerOutput:
        """This summarizer uses an LLM to summarize the text extract."""

        class LLMSummarizerOutput(BaseModel):
            guessed_questions: list[str]
            answer: str

        model_response = query_llm(
            messages=[
                {
                    "role": "system",
                    "content": self.summarizer_system_prompt,
                },
                {
                    "role": "user",
                    "content": self.summarizer_prompt.format(
                        text_extract=text_extract,
                    ),
                }
            ],
            temperature=self.temperature,
            model_name=self.summarizer_model_name,
            response_format=LLMSummarizerOutput,
        )

        if not isinstance(model_response, LLMSummarizerOutput):
            return SummarizerOutput(
                # Failed prediction if the LLM gives no answer.
                answer_pred=FAILED_PRED,
                summarizer_llm_response=None,
            )

        return SummarizerOutput(
            # Failed prediction if the LLM gives no answer.
            answer_pred=(
                model_response.answer
                if hasattr(model_response, "answer")
                else FAILED_PRED
            ),
            summarizer_llm_response=str(model_response),
            summarizer_llm_guessed_questions=(
                model_response.guessed_questions
                if hasattr(model_response, "guessed_questions")
                else None
            ),
        )


class HSStructuredHighlighter(HSBaseline):
    """Highlighter that uses structured output."""

    extractor_prompt: str = dedent(
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

    def __init__(self, *args, extractor_prompt: str | None = None, **kwargs) -> None:
        super().__init__(
            *args,
            extractor_prompt=extractor_prompt or self.extractor_prompt,
            **kwargs,
        )

    def call_highlighter(
        self, context_str: str, question_str: str
    ) -> HighlighterOutput:
        """This highlither uses structured output to extract text from the context."""

        class LLMOutput(BaseModel):
            answer: str
            text_extracts: list[str]

        model_response = query_llm(
            messages=[
                {
                    "role": "user",
                    "content": self.extractor_prompt.format(
                        context=context_str,
                        question_str=question_str,
                    ),
                }
            ],
            temperature=self.temperature,
            model_name=self.highlighter_model_name,
            response_format=LLMOutput,
        )

        if not isinstance(model_response, LLMOutput):
            return HighlighterOutput(highlighter_llm_response=None)

        # Nothing to highlight.
        if not model_response.answer or NOANSWER_PRED in model_response.answer:
            return HighlighterOutput(highlighter_llm_response=str(model_response))

        if not model_response.text_extracts:
            return HighlighterOutput(highlighter_llm_response=str(model_response))

        # Check if the text extracts are in the context.
        valid_text_extracts = []
        scores = []
        for text_extract in model_response.text_extracts:
            score, matched_text = fuzzmatch_extract(text_extract, context_str)
            scores.append(score)
            if matched_text and self._validate_text_extract(matched_text):
                valid_text_extracts.append(matched_text)

        valid_text = "\n".join(valid_text_extracts)

        return HighlighterOutput(
            highlighter_extracted=valid_text.strip(),
            highlighter_llm_response=model_response.answer,
            highlighter_text_extracts=model_response.text_extracts,
            highlighter_valid_text_extracts=valid_text_extracts,
            highlighter_fuzzmatch_scores=scores,
        )


class HSTwoStepsHighlighter(HSBaseline):
    """Highlighter that uses a two steps process:
    - answer the question
    - extract text based on the answer"""

    extractor_prompt: str = dedent(
        "You are an expert research assistant."
        "You are given a context text and an answer to a question about it. Your task is to extract all the "
        "relevant text extracts from the context that support the given answer."
        "Extract as many text extracts as needed to fully cover the answer."
        "The text extracts must be quoted _verbatim_!"
        "Context:\n"
        "{context}\n"
        "Question: {question_str}\n"
        "Answer: {answer}\n"
    )

    def __init__(self, *args, extractor_prompt: str | None = None, **kwargs) -> None:
        super().__init__(
            *args,
            extractor_prompt=extractor_prompt or self.extractor_prompt,
            **kwargs,
        )

    def call_highlighter(
        self, context_str: str, question_str: str
    ) -> HighlighterOutput:

        answer = query_llm(
            messages=[
                {
                    "role": "user",
                    "content": f"Answer the following question based on the provided context.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Question: {question_str}\n\n"
                    f"Answer:",
                }
            ],
            temperature=self.temperature,
            model_name=self.highlighter_model_name,
        )

        if not answer or NOANSWER_PRED in answer:
            return HighlighterOutput(highlighter_llm_response=answer)

        class LLMOutput(BaseModel):
            text_extracts: list[str]

        model_response = query_llm(
            messages=[
                {
                    "role": "user",
                    "content": self.extractor_prompt.format(
                        context=context_str,
                        question_str=question_str,
                        answer=answer,
                    ),
                }
            ],
            temperature=self.temperature,
            model_name=self.highlighter_model_name,
            response_format=LLMOutput,
        )

        if not isinstance(model_response, LLMOutput):
            return HighlighterOutput(highlighter_llm_response=None)

        if not model_response.text_extracts:
            return HighlighterOutput(highlighter_llm_response=str(model_response))

        # Check if the text extracts are in the context.
        valid_text_extracts = []
        scores = []
        for text_extract in model_response.text_extracts:
            score, matched_text = fuzzmatch_extract(text_extract, context_str)
            scores.append(score)
            if matched_text and self._validate_text_extract(matched_text):
                valid_text_extracts.append(matched_text)

        valid_text = "\n".join(valid_text_extracts)

        return HighlighterOutput(
            highlighter_extracted=valid_text.strip(),
            highlighter_llm_response=None,
            highlighter_text_extracts=model_response.text_extracts,
            highlighter_valid_text_extracts=valid_text_extracts,
            highlighter_fuzzmatch_scores=scores,
        )


class HSSpanHighlighter(HSBaseline):
    """Highlighter that uses structured output + span highlighting."""

    extractor_prompt: str = dedent(
        "I'd like for you to answer questions about a context text that will be provided."
        "I'll give you a pair with the form:\nContext: 'context text'\nQuestion: 'a question about the context'.\n"
        "First, tell me about your knowledge of the context and what information it contains, "
        "then, create an analysis of the context strictly using information contained in the text provided. "
        "If there is no information in the context that answers the question, "
        f"your answer _must_ be exactly '{NOANSWER_PRED}'.\n"
        "If the question can be answered, you must return the `answer` together with "
        "a list of text extracts (`text_extracts`) that allowed you to answer the question."
        "Rather than returning the text extracts directly, return just a few words surrounding the start and end."
        "For example, if the text extract is 'General relativity entails that gravity is the result of spacetime curvature', "
        "you might return start='General relativity' and end='spacetime curvature'.\n"
        "Here's the context and question for you to reason about and answer:\n"
        "Context:\n"
        "{context}\n"
        "Question: {question_str}?\n"
    )

    def __init__(self, *args, extractor_prompt: str | None = None, **kwargs) -> None:
        super().__init__(
            *args,
            extractor_prompt=extractor_prompt or self.extractor_prompt,
            **kwargs,
        )

    def call_highlighter(
        self, context_str: str, question_str: str
    ) -> HighlighterOutput:
        """This highlither uses structured output to extract text from the context."""

        class TextExtract(BaseModel):
            start: str
            end: str

        class LLMOutput(BaseModel):
            answer: str
            text_extracts: list[TextExtract]

        model_response = query_llm(
            messages=[
                {
                    "role": "user",
                    "content": self.extractor_prompt.format(
                        context=context_str,
                        question_str=question_str,
                    ),
                }
            ],
            temperature=self.temperature,
            model_name=self.highlighter_model_name,
            response_format=LLMOutput,
        )

        if not isinstance(model_response, LLMOutput):
            return HighlighterOutput(highlighter_llm_response=None)

        # Nothing to highlight.
        if not model_response.answer or NOANSWER_PRED in model_response.answer:
            return HighlighterOutput(highlighter_llm_response=str(model_response))

        if not model_response.text_extracts:
            return HighlighterOutput(highlighter_llm_response=str(model_response))

        # Validate and reconstruct text extracts.
        valid_text_extracts = []
        scores = []
        for text_extract in model_response.text_extracts:
            # Find start and end positions in the context
            start_pos = context_str.find(text_extract.start)
            end_pos = context_str.find(text_extract.end)

            if start_pos == -1 or end_pos == -1 or start_pos >= end_pos:
                # Invalid start/end markers
                scores.append(0)
                continue

            # Extract text from start to end (inclusive of end phrase)
            extracted_text = context_str[start_pos : end_pos + len(text_extract.end)]

            # Apply word count validation
            if self._validate_text_extract(extracted_text):
                valid_text_extracts.append(extracted_text)
            scores.append(100)  # Exact match by construction

        valid_text = "\n".join(valid_text_extracts)

        return HighlighterOutput(
            highlighter_extracted=valid_text.strip(),
            highlighter_llm_response=model_response.answer,
            highlighter_text_extracts=[
                te.start + "..." + te.end for te in model_response.text_extracts
            ],
            highlighter_fuzzmatch_scores=scores,
        )


class HSBERTExtractor(HSBaseline):
    """Highlighter that uses a BERT-based extractor."""

    def __init__(self, highlighter_threshold=0.3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Lazy loading.
        from transformers.models.auto.tokenization_auto import AutoTokenizer
        from transformers.pipelines.question_answering import QuestionAnsweringPipeline
        from transformers.models.auto.modeling_auto import AutoModelForQuestionAnswering

        try:
            model = AutoModelForQuestionAnswering.from_pretrained(self.highlighter_model_name)  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(self.highlighter_model_name)  # type: ignore

            # Move to GPU if available
            device = 0 if torch.cuda.is_available() else -1
            if device >= 0:
                model = model.cuda(device)

            # Create the pipeline directly
            self.extractor = QuestionAnsweringPipeline(  # type: ignore
                model=model,
                tokenizer=tokenizer,
                device=device,
                framework="pt",
                handle_impossible_answer=True,
                max_answer_len=100,
                max_question_len=100,
            )
        except OSError as e:
            raise ValueError(
                f"Failed to load the highlighter model '{self.highlighter_model_name}'. Is it there?"
            )
        self.highlighter_threshold = highlighter_threshold
        # In case this hasn't been downloaded yet.
        import nltk

        nltk.download("punkt_tab", quiet=True)

    def call_highlighter(
        self, context_str: str, question_str: str
    ) -> HighlighterOutput:
        """This highlighter uses a BERT-based extractor to extract text from the context."""
        try:
            model_response: dict = self.extractor(  # type: ignore
                question=question_str,
                context=context_str,
            )
        except Exception as e:
            print(f"Error in call_highlighter: {e}")
            return HighlighterOutput(highlighter_llm_response=f"Error: {e}")

        # Nothing to highlight.
        if (
            model_response["score"] < self.highlighter_threshold
            or not model_response["answer"].strip()
        ):
            return HighlighterOutput(
                highlighter_extracted=None,
                highlighter_score=model_response["score"],
                highlighter_llm_response=model_response["answer"],
            )

        # We return the full sentence that contains the answer.
        context = context_str.replace("\n", " ")
        for s in sentences(context):  # type: ignore
            if s in context[model_response["start"] :]:
                extracted_sentence = s
                break
        else:
            raise ValueError(
                "No sentence found containing the answer. This looks like a logical error in the code."
            )

        # Apply word count validation
        if not self._validate_text_extract(extracted_sentence):
            return HighlighterOutput(
                highlighter_extracted=None,
                highlighter_score=model_response["score"],
                highlighter_llm_response=model_response["answer"],
            )

        return HighlighterOutput(
            highlighter_extracted=extracted_sentence.strip(),
            highlighter_score=model_response["score"],
            highlighter_llm_response=model_response["answer"],
        )
