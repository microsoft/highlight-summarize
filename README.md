# Highlight then Summarize

This repository collects the code that enables reproducing out experiments for the H&S paper.

First, `pip install -r requirements.txt`.
The main experiments are configured via `experiments.yaml`, and they are run
via `python run_experiments.py`.
**NOTE**: the [judges](https://github.com/quotient-ai/judges) library that we use
didn't play too well with our multithreading implementation.
If you see an error such as "too many open files", this may be because `judges`
is opening several TCP connections, which are amplified by our multithreading setup,
and this could make the code fail.
A workaround is to `ulimit -n 10000` before running this code; in our experiments,
the number of TCP connections never exceeded 2000, so this should work fine.