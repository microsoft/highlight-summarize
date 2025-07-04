import os
import base64
import streamlit as st

CONTENTS_DIR = "contents"
def load_content(fname, mode="r"):
    """Load content from a file in the contents directory."""
    with open(os.path.join(CONTENTS_DIR, fname), mode) as f:
        return f.read()

# pdf = load_content("highlight_summarize.pdf", mode="rb")

# st.sidebar.download_button(label="Get the paper",
#                         data=pdf,
#                         file_name="highlight_summarize.pdf",
#                         mime='application/octet-stream'
# )


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