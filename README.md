# Highlight then Summarize

This repository collects the code that enables reproducing out experiments for the H&S paper.

To get started:

```
git clone https://github.com/microsoft/highlight-summarize
# Ideally, you want to set up a virtualenv at this point:
python3 -m venv .venv
source .venv/bin/activate
pip install .
```


## H&S Demo app

The demo app implements a chatbot that answers questions about H&S (based on our paper)
by implementing the H&S pattern.

It has additional requirements, which can be installed via:

```
pip install .[demo]
```

To run it, after installing the `pip` dependencies, run `bash run.sh`.

--------------------

**Trademarks** This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
