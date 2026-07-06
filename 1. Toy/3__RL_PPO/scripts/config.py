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

    RL_ROOT  = os.path.join(MAIN_DIR, "3__RL_PPO")
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

    ACTION_CHANGE_COEF = 0.2
    ACTION_COS_COEF = 30.0
    INIT_OD_PRIOR = [3, 1, 2, 2]

    APP = "extension"
                                                       
                                                                 
    APP_OPTIONS = ("extension", "baseline")
    APP = str(APP).strip().lower()
    if APP not in APP_OPTIONS:
        raise RuntimeError(f"APP must be one of {APP_OPTIONS}, got {APP!r}.")

    TRIAL = 5
    EPOCHS = 1500

    if APP == "extension":
        STATE_DIM = 1 + 9 + 176 + 2 * NUM_OD
    elif APP == "baseline":
        STATE_DIM = 1 + 9 + 176

    ALGORITHM = "PPO"
    HYPERPARAMS = {
        "PPO": dict(
            learning_rate=0.0003,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            verbose=0,
            n_steps=180,
            batch_size=90,
        )
    }

    NUM_ENVS = 20
    PLOT_DPI = 600
    TRIAL_TIMES_NAME = f"{APP}_trial_times.csv"
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
