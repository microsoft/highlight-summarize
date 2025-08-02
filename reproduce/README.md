# Paper experiments

This folder contains the code to enable reproducing the experiments contained in our paper.

## Plots and results analysis

We share the full responses of H&S and RAG pipelines, as well as the ratings provided by
LLM as judges. These results can be dowloaded via:

```
git lfs pull --exclude=""
```

and then analyzed via the `data-analysis.ipynb` notebook. This notebook also enables
recreating all our experiments' figures and tables.

If you want to re-run the experiments from scratch, follow the guide below.

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