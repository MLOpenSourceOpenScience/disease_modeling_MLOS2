import csv
from typing import Optional, Iterable

import numpy as np


def save_np_as_csv(
    dest: str, arr: np.typing.NDArray, columns: Optional[Iterable] = None
):
    arr = arr.tolist()
    with open(dest, "w", newline="") as f:
        w = csv.writer(f, delimiter=",")
        if columns:
            w.writerow(columns)
        w.writerows(arr)
