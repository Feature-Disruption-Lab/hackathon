"""Helper functions"""

from typing import Dict, List

import numpy as np


def save_as_npz(data: List[Dict], filename: str):
    arrays_dict = {}
    for i, sample in enumerate(data):
        arrays_dict[f"prompt_{i}"] = sample["prompt"]
        arrays_dict[f"response_{i}"] = sample["response"]
        arrays_dict[f"dense_features_{i}"] = sample["features"]
        arrays_dict[f"label_{i}"] = sample["label"]
    np.savez(filename, **arrays_dict)


def load_as_npz(filename: str):
    loaded = np.load(filename, allow_pickle=True)
    data = []
    i = 0
    while f"response_{i}" in loaded:
        data.append(
            {
                "prompt": str(),
                "response": str(loaded[f"response_{i}"]),
                "features": loaded[f"dense_features_{i}"][None][0].toarray(),
                "label": str(loaded[f"label_{i}"]),
            }
        )
        i += 1
    return data

import time
from typing import Callable, TypeVar

T = TypeVar("T")

def retry_with_backoff(func: Callable[[],T], max_retries: int) -> T:
    retries = 0
    while True:
        try:
            return func()
        except Exception as e:
            if "rate limit" in str(e).lower():
                if retries < max_retries:
                    sleep_time = 2**retries
                    print(
                        f"Rate limit encountered. Retrying in {sleep_time} seconds..."
                    )
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    print(f"Max retries reached for function {func}.")
                    raise e
            else:
                raise e
