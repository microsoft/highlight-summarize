# Paper experiments

This folder contains the code to enable reproducing the experiments contained in our paper.

There's two ways of using this code:
- we share all the artifacts from our evaluation in `results/`, which means you can go ahead and use them directly for analysis (see "Plots and results analysis")
- you want to rerun it all from scratch; this will produce new `results/` ("Re-run from scratch").

## Plots and results analysis

We share the full responses of H&S and RAG pipelines, as well as the ratings provided by
LLM as judges. These results can be dowloaded via:

```
git lfs pull --exclude=""
```

(The above assumes you have installed `git-lfs`. E.g., `sudo apt install git-lfs`)

They can then be found under the `results/` folder.

We provide a notebook, `data-analysis.ipynb`, which enables
recreating all our experiments' figures and tables.

## Re-run from scratch

If instead you want to re-run the experiments from scratch, follow these steps.
You will want to delete `results/`, or otherwise the script will think you've already run all the experiments and it won't do anything.

### Main experiments (inference + judgement)

The main experiments are configured via `experiments.yaml`, and they are run
via `python run_experiments.py` after installing all the requirements (see the main repo's README.md for instructions).

**NOTE**: the [judges](https://github.com/quotient-ai/judges) library that we use
didn't play too well with our multithreading implementation.
If you see an error such as "too many open files", this may be because `judges`
is opening several TCP connections, which are amplified by our multithreading setup,
and this could make the code fail.
A workaround is to run `ulimit -n 10000` before running this code; in our experiments,
the number of TCP connections never exceeded 2000, so this should work fine.

After running this script, you should be able to run part of the jupyter notebook. (For the other part, look at the next sections.)

**If** you want to obtain `.csv` files, similar to the ones we committed to the repo, use `results-to-csv.py`.
However, this is only necessary if you want to remove any data from the HF datasets and make the files smaller.

### Pairwise comparisons

After running `run_experiments.py`, we can do pairwise comparisons via an LLM
as a judge.
In particular:
- compare between highlighter vs HS: `python compare.py highlighter <run_folder>`. E.g., `python compare.py highlighter results/repliqa_3/HSBaseline-gpt-4.1-mini-gpt-4.1-mini`.
- compare the pipelines 1-to-1: `python compare.py pairwise <dataset results folder>`. E.g., `python compare.py pairwise results/repliqa_3`.

**Note:** you'll need to remove the respective `comparison-` files under `results/` before running this, or your experiments may be skipped.

After running these, you'll be able to look at the results via the jupyter notebook.

### DeBERTaV3 fine-tuning

In our experiments, we also used a DeBERTaV3 model as highlighter, fine-tuned on the RepliQA dataset.
To repeat this fine-tuning, run `python qa-extractor-finetune.py`.