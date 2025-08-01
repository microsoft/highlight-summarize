import sys
import streamlit as st
from pydantic import BaseModel

from utils import sidebar, load_content, log
<<<<<<< HEAD
from highlight_summarize.hs import HSStructuredHighlighter, HSBaselinePrediction
=======
sys.path.append("..")
from src.hs import HSStructuredHighlighter, HSBaselinePrediction
>>>>>>> cf444e549c1995aac546dc7a5040358c86aac1d3

#################################
# H&S and data.
#################################
hs = HSStructuredHighlighter(
        highlighter_model_name="gpt-4.1-mini",
        summarizer_model_name="gpt-4.1-mini",
        min_highlighted_words=15,
    )

# "RAG" document.
doc = load_content("highlight_summarize.txt").strip()
if not doc.strip:
    raise ValueError("The document is empty. Please provide a valid document.")
detective = load_content("detective.png", mode="rb")


#################################
# Page.
#################################
st.title("💬 Highlight&Summarize")
st.caption("🚀 A demo of H&S, which can answer questions about H&S!")
sidebar()
# show_internals = st.toggle("Show the internal process of H&S", value=False)
show_internals = True

class Msg(BaseModel):
    role: str
    content: str
    full_response: HSBaselinePrediction | None = None

    def display(self):
        """Display the message in the chat."""
        if self.full_response and show_internals:
            self.display_intermediate()
        st.chat_message(self.role).write(self.content)

    def display_intermediate(self):
        """Display the intermediate outputs of the H&S model."""
        if not self.full_response:
            raise ValueError("No full response to display.")
        md = "Psst! Here's what is happening inside H&S:\n\n"
        texts_and_scores = "\n".join([
            f"{text} (Score: {score})"
            for text, score in zip(self.full_response.highlighter_text_extracts, self.full_response.highlighter_text_extracts_scores)
        ])
        lines = [
            f"**Highlighter Output:** {texts_and_scores}.",
            f"**What question the summarizer thinks was asked:** {self.full_response.summarizer_llm_guessed_question}.",
            f"**Summarizer response:** {self.full_response.answer_pred}."
        ]
        for line in lines:
            md += f":blue[{line}]\n\n"
        with st.expander("See the intermediate outputs in H&S", expanded=False):
            st.chat_message(name="detective", avatar=detective).write(md)


# Initialize session state.
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        Msg(role="assistant",
            content="Hi! I can answer questions about H&S. If there's something I don't know, I'll just say so. How can I help you?")
    ]

# Display chat history.
for msg in st.session_state.messages:
    msg.display()

if prompt := st.chat_input():
    msg = Msg(role="user", content=prompt)
    msg.display()
    st.session_state.messages.append(msg)

    response = hs.call_model(
        context_str=doc,
        question_str=prompt
    )

    # Handle the response.
    if not response.answer_pred or response.answer_pred == "UNANSWERABLE":
        answer = "I don't know the answer to that question. I can only answer based on my knowledge about H&S."
    else:
        answer = response.answer_pred
    msg = Msg(role="assistant", content=answer, full_response=response)
    msg.display()
    st.session_state.messages.append(msg)

    # Logging.
    log({
        "question": prompt,
        "answer": response.answer_pred,
        "response": response.model_dump(),
    })