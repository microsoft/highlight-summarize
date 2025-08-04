# Highlight & Summarize

This repository contains code for reproducing the experiments described in the H&S paper and running the H&S demo.



## Getting Started

```
# Clone the git repository
git clone https://github.com/microsoft/highlight-summarize

# Create a Python virtual environment
python3 -m venv .venv

# Activate the virtual environment (Linux)
source .venv/bin/activate

# Activate the virtual environment (Windows)
.venv\Scripts\Activate.ps1 # On Windows

# Install this project and its dependencies 
pip install .[all]
```

To reproduce the experiments, head over to [reproduce](reproduce/).

## H&S Demo app

<img width="741" height="599" alt="image" src="https://github.com/user-attachments/assets/4334aac5-8e3c-44da-a02a-e704423a06d0" />


The demo app implements a chatbot that answers questions about H&S (based on our paper)
by implementing the H&S pattern.

It has additional requirements. To run it:

```
pip install .[demo]
cd demo
bash run.sh
```
**Note** By default, the demo app listens on `0.0.0.0`. You may want to change this to `localhost` by editing `demo/run.sh`.

--------------------

**Trademarks** This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
