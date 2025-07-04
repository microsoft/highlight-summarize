import os

CONTENTS_DIR = "demo/contents"

def load_content(fname, mode="r"):
    """Load content from a file in the contents directory."""
    with open(os.path.join(CONTENTS_DIR, fname), mode) as f:
        return f.read()