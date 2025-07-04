import os
import base64
import streamlit as st

CONTENTS_DIR = "contents"
def load_content(fname, mode="r"):
    """Load content from a file in the contents directory."""
    with open(os.path.join(CONTENTS_DIR, fname), mode) as f:
        return f.read()

with st.sidebar:
    pdf = load_content("highlight_summarize.pdf", mode="rb")

    st.download_button(label="Get the paper",
                        data=pdf,
                        file_name="highlight_summarize.pdf",
                        mime='application/octet-stream'
    )
    # openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    # "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"
    # "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
    # "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"



def img_to_bytes(img_path):
    img_bytes = load_content(img_path, mode="rb")
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='max-width: 90%'>".format(
      img_to_bytes(img_path)
    )
    return img_html

md = load_content("about.md")
st.markdown(md.replace("{hs_image}", img_to_html("hs.png")), unsafe_allow_html=True)