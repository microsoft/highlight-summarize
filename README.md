# Highlight then Summarize

This repository collects the code that enables reproducing out experiments for the H&S paper.

First, `pip install -r requirements.txt`.

## Main experiments (inference + judgement)

The main experiments are configured via `experiments.yaml`, and they are run
via `python run_experiments.py`.
**NOTE**: the [judges](https://github.com/quotient-ai/judges) library that we use
didn't play too well with our multithreading implementation.
If you see an error such as "too many open files", this may be because `judges`
is opening several TCP connections, which are amplified by our multithreading setup,
and this could make the code fail.
A workaround is to `ulimit -n 10000` before running this code; in our experiments,
the number of TCP connections never exceeded 2000, so this should work fine.

## Pairwise comparisons

After running `run_experiments.py`, we can do pairwise comparisons via an LLM
as a judge.
In particular:
- compare between highlighter vs HS: `python compare.py highlighter <run_folder>`. E.g., `python compare.py highlighter results/repliqa_3/HSBaseline-gpt-4.1-mini-gpt-4.1-mini`.
- compare the pipelines pairwise: `python compare.py pairwise <dataset results folder>`. E.g., `python compare.py pairwise results/repliqa_3`.

## H&S Demo app

To run it, after installing the `pip` dependencies, run `bash run.sh`.