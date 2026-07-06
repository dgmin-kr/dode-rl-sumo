import csv
import os
from datetime import timedelta


FIELDNAMES = ("elapsed_sec", "elapsed_hms")
FILENAME = "run_time.csv"


def _path(result_dir: str, filename: str = FILENAME) -> str:
    return os.path.join(result_dir, filename)


def reset_run_time(result_dir: str, filename: str = FILENAME) -> str:
    os.makedirs(result_dir, exist_ok=True)
    path = _path(result_dir, filename)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
    return path


def write_run_time(result_dir: str, elapsed_sec: float, filename: str = FILENAME) -> str:
    os.makedirs(result_dir, exist_ok=True)
    path = _path(result_dir, filename)
    if not os.path.exists(path):
        reset_run_time(result_dir, filename)
    elapsed_sec = float(elapsed_sec)
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writerow({
            "elapsed_sec": f"{elapsed_sec:.6f}",
            "elapsed_hms": str(timedelta(seconds=round(elapsed_sec))),
        })
    return path
