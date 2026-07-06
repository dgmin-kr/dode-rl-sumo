import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "sumo_patch.py").is_file()), None)
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import Config
from env import MySumoEnv


_NEEDED_WORKER_FILES = ("detectors.add.xml", "network.net.xml", "run.sumocfg")
_APP_STATE_DIM = {
    "extension": 1 + 9 + 176 + 2 * Config.NUM_OD,
    "baseline": 1 + 9 + 176,
}


def _normalize_app(value: str) -> str:
    app = str(value).strip().lower()
    if app not in _APP_STATE_DIM:
        raise ValueError(f"app must be one of {tuple(_APP_STATE_DIM)}, got {value!r}")
    return app


def _load_records(best_json_path: Path):
    text = best_json_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] in "[{":
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

    records = []
    for line in text.splitlines():
        s = line.strip().rstrip(",")
        if s:
            records.append(json.loads(s))
    return records


def load_action_plans(result_root: Path, source_app: str):
    pattern = re.compile(rf"{re.escape(source_app)}_trial(\d+)")
    trial_dirs = []
    for path in result_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.fullmatch(path.name)
        if match:
            trial_dirs.append((int(match.group(1)), path))
    trial_dirs.sort(key=lambda x: x[0])

    if not trial_dirs:
        raise FileNotFoundError(f"Could not find '{source_app}_trial*' folders: {result_root}")

    actions_2d_list = []
    trial_names = []
    for _, trial_dir in trial_dirs:
        fp = trial_dir / "best_trajectory.json"
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {fp}")

        records = _load_records(fp)
        if not records:
            raise ValueError(f"Record is empty: {fp}")

        actions = np.asarray([rec["action"] for rec in records], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(f"action must be two-dimensional: {fp}, shape={actions.shape}")

        actions_2d_list.append(actions)
        trial_names.append(trial_dir.name)

    shapes = [a.shape for a in actions_2d_list]
    if len(set(shapes)) != 1:
        raise ValueError(f"Cannot stack actions because trial action shapes differ: {dict(zip(trial_names, shapes))}")

    actions_3d = np.stack(actions_2d_list, axis=0)
    print("trial_names:", trial_names)
    print("2D shapes:", shapes)
    print("3D shape:", actions_3d.shape)
    return trial_names, actions_3d


def prepare_worker_dir(worker_idx=0):
    Config.ensure_dirs()
    src = Config.BASE_WORK_DIR
    dst = Config.worker_dir(worker_idx)
    os.makedirs(dst, exist_ok=True)
    for name in _NEEDED_WORKER_FILES:
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"No utils file: {src_path}")
        shutil.copy2(src_path, dst_path)
    os.makedirs(os.path.join(dst, "dump"), exist_ok=True)
    return dst


def remove_workers_root():
                                                            
    root = os.path.abspath(Config.WORKERS_ROOT)
    project_root = os.path.abspath(Config.RL_ROOT)
    if os.path.basename(root) != "workers":
        raise RuntimeError(f"Refusing to delete non-workers path: {root}")
    if os.path.commonpath([root, project_root]) != project_root:
        raise RuntimeError(f"Refusing to delete path outside experiment root: {root}")
    if os.path.isdir(root):
        shutil.rmtree(root)

def build_env(worker_idx=0, total_step=None, seed=42, app="baseline"):
    app = _normalize_app(app)
    rl_dir = prepare_worker_dir(worker_idx)
    answer_path = Config.answer_path_for(worker_idx)
    sumocfg_path = os.path.join(rl_dir, "run.sumocfg")

    if not os.path.exists(rl_dir):
        raise FileNotFoundError(f"No directory: {rl_dir}")
    if not os.path.exists(sumocfg_path):
        raise FileNotFoundError(f"No run.sumocfg: {sumocfg_path}")
    if not os.path.exists(answer_path):
        raise FileNotFoundError(f"No answer.csv: {answer_path}")

    if total_step is None:
        total_step = Config.TOTAL_STEP

    env = MySumoEnv(
        rl_dir=rl_dir,
        sumo_binary=Config.SUMO_BINARY,
        origin_list=Config.ORIGIN_LIST,
        destination_list=Config.DESTINATION_LIST,
        input_interval=Config.INPUT_INTERVAL,
        detector_interval=Config.DETECTOR_INTERVAL,
        num_OD=Config.NUM_OD,
        num_det=Config.NUM_DET,
        state_dim=_APP_STATE_DIM[app],
        answer_dir=answer_path,
        total_step=int(total_step),
        seed=seed,
        action_change_coef=Config.ACTION_CHANGE_COEF,
        action_cos_coef=Config.ACTION_COS_COEF,
        init_od_prior=Config.INIT_OD_PRIOR,
        app=app,
    )
    obs, info = env.reset(seed=seed)
    return env, obs, info


def run(actions, trial=0, worker_idx=0, seed=42, out_path=None, app="baseline"):
    total_step = int(Config.TOTAL_STEP)
    env, obs, info = build_env(worker_idx=worker_idx, total_step=total_step, seed=seed, app=app)

    total_reward = 0.0
    last_info = info
    last_step = -1

    try:
        for t in range(total_step):
            action_t = actions[t]
            obs, reward, terminated, truncated, info = env.step(action_t)
            total_reward += float(reward)
            last_info = info
            last_step = t

            if terminated or truncated:
                break
    finally:
        env.close()

    trajectory = last_info.get("trajectory", [])

    if out_path is None:
        os.makedirs(Config.RESULT_DIR, exist_ok=True)
        out_path = os.path.join(Config.RESULT_DIR, f"trajectory_{trial}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)

    return trajectory, total_reward, last_step, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-app", default="baseline", choices=tuple(_APP_STATE_DIM))
    parser.add_argument("--replay-app", default=None, choices=tuple(_APP_STATE_DIM))
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--worker-idx", type=int, default=0)
    args = parser.parse_args()

    source_app = _normalize_app(args.source_app)
    replay_app = _normalize_app(args.replay_app or source_app)
    result_root = Path(Config.RESULT_DIR)

    trial_names, actions_3d = load_action_plans(result_root=result_root, source_app=source_app)
    num_trials = len(trial_names) if args.num_trials is None else min(int(args.num_trials), len(trial_names))

    for trial in range(num_trials):
        seed_trial = trial
        out_json = os.path.join(Config.RESULT_DIR, f"trajectory_{trial}.json")

        traj, score, end_step, used_plan = run(
            actions=actions_3d[trial],
            trial=trial,
            worker_idx=args.worker_idx,
            seed=seed_trial,
            out_path=out_json,
            app=replay_app,
        )

        print(
            f"[trial {trial}] source_app={source_app}, replay_app={replay_app}, "
            f"total_reward={score:.6f}, last_step={end_step}, saved={out_json}, traj_len={len(traj)}"
        )

    remove_workers_root()

if __name__ == "__main__":
    main()
