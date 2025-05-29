from rapidfuzz import fuzz
from textwrap import dedent
from types import FunctionType
from pydantic import BaseModel

from qa import QAEvaluator

class SummarizerOutput(BaseModel):
    guessed_question: str
    answer: str

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
            """You are an expert research assistant.
            You are tasked with highlighting relevant text extract(s) from a context text.
            Your output must be (verbatim) a list of the text extract(s) that answer the question.
            You will output the text extract(s), by prefixing each extract with a bullet point "-",
            and nothing else.
            Important: if no part of the text answers the question, you must output "UNANSWERABLE".
            Context:
            {context}
            Question: {question_str}?\n
            """)
        self.summarizer_prompt = dedent(
            """You are given highlighted text from a document; the text extract is relevant to some
            question which you don't know. Figure out what question the text extract is trying
            to answer, and summarize the text extract in a concise manner in the form of an answer.
            Text extract:
            {text_extract}
            """)

    def call_model(self, context_str: str, question_str: str) -> tuple[str | None, str | None]:
        # NOTE: we can't be too liberal with the metadata fields here: we need
        # to specify them in advance, or the HF Dataset will fail to create.
        metadata_fields = ["raw_response", "guessed_question", "raw_text_extracts"]
        metadata = {field: None for field in metadata_fields}
        highlighted, highliter_meta = self.call_highlighter(context_str, question_str)
        metadata.update(highliter_meta)

        if not highlighted:
            return "UNANSWERABLE", metadata

        answer, summarizer_meta = self.call_summarizer(highlighted)
        metadata.update(summarizer_meta)
        return answer, metadata
        
    def call_highlighter(self, context_str: str, question_str: str) -> tuple[str | None, dict[str, str]]:
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
        # Parse.
        if not model_response.choices or not model_response.choices[0].message.content:
            return None, {}
        if not model_response.choices[0].message.content.startswith("- "):
            # print(f"Unexpected response format: {model_response.choices[0].message.content}")
            return None, {"raw_response": model_response.choices[0].message.content}
        # Extract the text extract(s) from the response.
        text_extracts = model_response.choices[0].message.content.strip().split("\n")
        if "UNANSWERABLE" in text_extracts:
            return None, {"raw_response": model_response.choices[0].message.content}
        # Remove the bullet points and strip whitespace.
        text_extracts = [extract.strip().lstrip("- ") for extract in text_extracts if extract.strip()]
        if not text_extracts:
            # print("No text extracts found in the response.")
            return None, {"raw_response": model_response.choices[0].message.content}
        
        # Check if the text extracts are in the context.
        valid_text_extracts = []
        for text_extract in text_extracts:
            score = fuzz.partial_ratio(text_extract, context_str)
            if score < 95:
                    print(f"Score: {score}")
                    print(f"Text extract not found in context: {text_extract}")
                    print(f"Context: {context_str}")
            else:
                valid_text_extracts.append(text_extract)
        
        valid_text = "\n".join(valid_text_extracts)

        return valid_text.strip() if valid_text else None, {"raw_text_extracts": text_extracts}

    def call_summarizer(self, text_extract: str) -> str:
        # Structured output: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/structured-outputs?tabs=python-secure%2Cdotnet-entra-id&pivots=programming-language-python.
        raw_response = self.openai_client().beta.chat.completions.parse(
            messages=[
                {"role": "user", "content": self.summarizer_prompt.format(
                    text_extract=text_extract,
                )}
            ],
            temperature=self.temperature,
            model=self.model_name,
            response_format=SummarizerOutput,
        ).choices[0].message.parsed

        if not raw_response.answer:
            print(f"Empty answer from summarizer: {raw_response}")
            return "UNANSWERABLE", {"raw_response": raw_response}
        if not hasattr(raw_response, "guessed_question") or not raw_response.guessed_question:
            print(f"Empty guessed question from summarizer: {raw_response}")
            return raw_response.answer, {"raw_response": raw_response}
        
        return raw_response.answer, {"guessed_question": raw_response.guessed_question}