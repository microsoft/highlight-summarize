"""Enables threaded execution of GPT-like calls."""

import time
import datasets
import concurrent
from tqdm import tqdm

MAX_THREADS = 50


class Counter:
    """Monitors the status of our threads. Used for debugging and monitoring."""

    running = 0
    waiting = 0
    failed = 0

    def __init__(self, pbar):
        self.pbar = pbar
        self.display()

    def display(self):
        self.pbar.set_description(
            f"Running: {self.running}, Waiting: {self.waiting}, Failed: {self.failed}. Progress"
        )

    def update(self, waiting=0, running=0, failed=0):
        self.waiting += waiting
        self.running += running
        self.failed += failed
        self.display()


def mt_exec(example: dict, function: callable) -> dict:
    """Contains all the thread logic for launching the function, including sleeping and error handling.

    If it fails, it waits 2x the time it waited before, until a limit of >32
    (i.e., when ~1 minute has passed).
    """
    global counter
    if not "counter" in globals():
        counter = Counter(tqdm(disable=True))
    counter.update(running=1)

    wait_seconds = 1
    error = None
    while wait_seconds <= 32:
        try:
            res = function(example)
            counter.update(running=-1)
            return res
        except Exception as e:
            pass

        # Wait.
        counter.update(waiting=1)
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter.update(waiting=-1)

    # We were killed by the rate limit. Won't try anymore.
    print("Failed after trying for more than 1 minute.")
    if error:
        print(f"Last error: {error}")
    counter.update(failed=1, running=-1)
    return {}


def mt_map(
    function: callable, dataset: datasets.Dataset, max_threads: int = MAX_THREADS
) -> datasets.Dataset:
    """Maps a function over a dataset using multiple threads."""
    global counter  # Gives us fancy stats and a progress bar.

    # Map to dict.
    dataset_dict = {ex_id: example for ex_id, example in enumerate(dataset.to_list())}

    with tqdm(total=len(dataset)) as pbar:
        # Global counter for threads stats data.
        counter = Counter(pbar)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            # Start the load operations and mark each future with its example id.
            future_to_example = {
                executor.submit(
                    mt_exec,
                    example=dataset_dict[ex_id],
                    function=function,
                ): ex_id
                for ex_id in dataset_dict
            }
            for future in concurrent.futures.as_completed(future_to_example):
                ex_id = future_to_example[future]
                try:
                    res = future.result()
                except Exception as e:
                    print(f"Failed{ex_id}: {e}")
                else:
                    dataset_dict[ex_id].update(res)
                pbar.update(1)

    # Convert back to a Dataset.
    return datasets.Dataset.from_list(list(dataset_dict.values()))
