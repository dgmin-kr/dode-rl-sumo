import os
import shutil
import sys
from pathlib import Path
from copy import deepcopy

_PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "sumo_patch.py").is_file()), None)
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from sumo_patch import get_sumo_binary

class Config:

    MAIN_DIR = str(Path(__file__).resolve().parents[2])
    TIMING_UTILS_DIR = os.path.join(MAIN_DIR, "utils")
    if TIMING_UTILS_DIR not in sys.path:
        sys.path.insert(0, TIMING_UTILS_DIR)
    SUMO_BINARY = get_sumo_binary(__file__)

    RL_ROOT  = os.path.join(MAIN_DIR, "5__ST_BO (5sec)")
    RESULT_DIR     = os.path.join(RL_ROOT, "result")
    BASE_WORK_DIR  = os.path.join(MAIN_DIR, "utils")
    WORKERS_ROOT   = os.path.join(RL_ROOT, "workers")
    ANSWER_NAME = "answer.csv"
    ANSWER_DIR = os.path.join(MAIN_DIR, "1__Ground_truth")
    ANSWER_PATH = os.path.join(ANSWER_DIR, ANSWER_NAME)

    INPUT_INTERVAL     = 5
    DETECTOR_INTERVAL  = 300
    TOTAL_TIME         = 30
    TOTAL_STEP         = (TOTAL_TIME * 60) // INPUT_INTERVAL

    ORIGIN_LIST      = ["N1", "N4"]
    DESTINATION_LIST = ["N2", "N3"]
    NUM_OD  = 4
    NUM_DET = 9

    PLOT_DPI = 600
    CSV_NAME = "episode_rewards.csv"
    PNG_NAME = "episode_rewards.png"

    @staticmethod
    def worker_dir(idx: int) -> str:
        return os.path.join(Config.WORKERS_ROOT, f"worker_{idx}")

    @staticmethod
    def answer_path_for(idx: int) -> str:
        return Config.ANSWER_PATH

    @staticmethod
    def ensure_dirs():
        Path(Config.RESULT_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.WORKERS_ROOT).mkdir(parents=True, exist_ok=True)
