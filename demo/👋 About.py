import base64
import streamlit as st

from utils import sidebar, load_content



def img_to_bytes(img_path):
    img_bytes = load_content(img_path, mode="rb")
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid' style='max-width: 90%'>".format(
      img_to_bytes(img_path)
    )
    return img_html

st.set_page_config(
    page_title="Highlight & Summarize",
    page_icon="💬",
)

sidebar()
md = load_content("about.md")
st.markdown(md.replace("{hs_image}", img_to_html("hs.png")), unsafe_allow_html=True)