import os
import sys
import re
from pathlib import Path
from typing import List, Sequence, Tuple, Optional

import numpy as np

_PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "sumo_patch.py").is_file()), None)
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from sumo_patch import get_sumo_binary

class Config:

    MAIN_DIR = str(Path(__file__).resolve().parents[2])
    TIMING_UTILS_DIR = os.path.join(str(Path(MAIN_DIR).parent), "utils")
    if TIMING_UTILS_DIR not in sys.path:
        sys.path.insert(0, TIMING_UTILS_DIR)
    SUMO_BINARY = get_sumo_binary(__file__)

    RL_ROOT  = os.path.join(MAIN_DIR, "1__RL_PPO")
    RESULT_DIR     = os.path.join(RL_ROOT, "result")
    BASE_WORK_DIR  = os.path.join(RL_ROOT, "utils")
    WORKERS_ROOT   = os.path.join(RL_ROOT, "workers")
    ANSWER_NAME = "answer.csv"

    ACTION_UNIT_VEH    = 1
    INPUT_INTERVAL     = 5
    DETECTOR_INTERVAL  = 300
    TOTAL_TIME         = 30
    TOTAL_STEP         = (TOTAL_TIME * 60) // INPUT_INTERVAL

    ORIGIN_LIST      = ["Z2", "Z3", "Z7", "Z10", "Z12", "Z14", "Z16"]
    DESTINATION_LIST = ORIGIN_LIST
    NUM_OD  = 42

    ids = ['402364', '400971', '409528', '422357']
    DETECTOR_BASE_IDS = sorted(ids)
    NUM_DET = len(ids)

    DETECTOR_ID_SEP = "_"

    DETECTOR_SORT_BASE_IDS = True

    DETECTOR_STRICT = True

    DETECTOR_SORT_LANES = True

    NUM_DET = len(DETECTOR_BASE_IDS)
    EPOCHS = 15000
    INIT_ACTION_PROB = 0.5

    STATE_DIM = 1 + 1542 + NUM_DET

    ALGORITHM = "PPO"
    HYPERPARAMS = {
        "PPO": dict(
            learning_rate=0.0002,
            gamma=0.999,
            gae_lambda=0.99,
            ent_coef=0.001,
            verbose=0,
            n_steps=720,
            batch_size=720,
        )
    }

    NUM_ENVS = 30
    PLOT_DPI = 600
    CSV_NAME = "episode_rewards.csv"
    PNG_NAME = "episode_rewards.png"

    @staticmethod
    def worker_dir(idx: int) -> str:
        return os.path.join(Config.WORKERS_ROOT, f"worker_{idx}")

    @staticmethod
    def answer_path_for(idx: int) -> str:
        return os.path.join(Config.worker_dir(idx), Config.ANSWER_NAME)

    @staticmethod
    def ensure_dirs():
        Path(Config.RESULT_DIR).mkdir(parents=True, exist_ok=True)
        Path(Config.WORKERS_ROOT).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def build_detectors_and_groups(
        det_id_list: Sequence[str],
        base_ids: Optional[Sequence[str]] = None,
        sep: Optional[str] = None,
        sort_base_ids: Optional[bool] = None,
        sort_lanes: Optional[bool] = None,
    ) -> Tuple[List[str], List[Tuple[int, int]], List[str]]:
        if base_ids is None:
            base_ids = Config.DETECTOR_BASE_IDS
        if sep is None:
            sep = Config.DETECTOR_ID_SEP
        if sort_base_ids is None:
            sort_base_ids = Config.DETECTOR_SORT_BASE_IDS
        if sort_lanes is None:
            sort_lanes = Config.DETECTOR_SORT_LANES

        base_ids_sorted = list(base_ids)
        if sort_base_ids:
            base_ids_sorted = sorted(base_ids_sorted)

        det_id_list = list(det_id_list)

        def lane_key(did: str, base: str) -> Tuple[int, str]:
            if did == base:
                return (0, did)
            prefix = f"{base}{sep}"
            if did.startswith(prefix):
                rest = did[len(prefix):]
                tok = rest.split(sep, 1)[0]
                m = re.match(r"^(\d+)", tok)
                if m:
                    return (int(m.group(1)), did)
            return (10**9, did)

        selected_ids: List[str] = []
        group_slices: List[Tuple[int, int]] = []

        for base in base_ids_sorted:
            if sep in base:
                ids_for_base = [did for did in det_id_list if did == base]
            else:
                prefix = f"{base}{sep}"
                ids_for_base = [did for did in det_id_list if (did == base) or did.startswith(prefix)]

            if sort_lanes:
                ids_for_base = sorted(ids_for_base, key=lambda did: lane_key(did, base))

            s = len(selected_ids)
            selected_ids.extend(ids_for_base)
            e = len(selected_ids)
            group_slices.append((s, e))

        return selected_ids, group_slices, base_ids_sorted

    @staticmethod
    def aggregate_detector_vector(det: Sequence[float], group_slices: Sequence[Tuple[int, int]]) -> np.ndarray:
        x = np.asarray(det, dtype=float).reshape(-1)
        y = np.empty(len(group_slices), dtype=float)
        for i, (s, e) in enumerate(group_slices):
            seg = x[s:e]
            seg2 = seg[np.isfinite(seg)]
            y[i] = seg2.sum() if seg2.size > 0 else np.nan
        return y
