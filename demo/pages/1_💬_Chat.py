import sys
import streamlit as st

from utils import sidebar, load_content, log
sys.path.append("..")
from src.hs import HSStructuredHighlighter

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


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi! I can answer questions about H&S. If there's something I don't know, I'll just say so. How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = hs.call_model(
        context_str=doc,
        question_str=prompt
    )

    # Display.
    md = "Psst! Here's what is happening inside H&S:\n\n"
    lines = [
        f"**Highlighter Output:** {response.highlighter_extracted}.",
        f"**What question the summarizer thinks was asked:** {response.summarizer_llm_guessed_question}.",
        f"**Summarizer response:** {response.answer_pred}."
    ]
    for line in lines:
        md += f":blue[{line}]\n\n"
    with st.expander("See the intermediate outputs in H&S", expanded=False):
        st.chat_message(name="detective", avatar=detective).write(md)
    st.session_state.messages.append({"role": "detective", "content": md})

    if not response.answer_pred or response.answer_pred == "UNANSWERABLE":
        st.chat_message("assistant").write("I don't know the answer to that question. I can only answer based on my knowledge about H&S.")
        st.session_state.messages.append({"role": "assistant", "content": "I don't know the answer to that question."})
    else:
        st.chat_message("assistant").write(response.answer_pred)
        st.session_state.messages.append({"role": "assistant", "content": response.answer_pred})

    # Logging.
    log({
        "question": prompt,
        "answer": response.answer_pred,
        "response": response.model_dump(),
    })