import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from config import Config
from env import MySumoEnv


_NEEDED_WORKER_FILES = ("answer.csv", "detectors.add.xml", "network.net.xml", "run.sumocfg")


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


def load_action_plan(result_root: Path):
    fp = result_root / "best_trajectory.json"
    if not fp.exists():
        raise FileNotFoundError(f"File not found: {fp}")

    records = _load_records(fp)
    if not records:
        raise ValueError(f"Record is empty: {fp}")

    actions = np.asarray([rec["action"] for rec in records], dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"action must be two-dimensional: {fp}, shape={actions.shape}")
    print("action shape:", actions.shape)
    return actions


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


def build_env(worker_idx=0, total_step=None, seed=42):
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
        state_dim=1,
        answer_dir=answer_path,
        total_step=int(total_step),
        action_unit_veh=Config.ACTION_UNIT_VEH,
    )
    obs, info = env.reset(seed=seed)
    return env, obs, info


def run(actions, run_idx=0, worker_idx=0, seed=42, out_path=None):
    total_step = int(Config.TOTAL_STEP)
    env, obs, info = build_env(worker_idx=worker_idx, total_step=total_step, seed=seed)

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
        out_path = os.path.join(Config.RESULT_DIR, f"trajectory_{run_idx}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, ensure_ascii=False, indent=2)

    return trajectory, total_reward, last_step, actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=None)
    parser.add_argument("--worker-idx", type=int, default=0)
    args = parser.parse_args()

    result_root = Path(Config.RESULT_DIR)
    actions = load_action_plan(result_root=result_root)
    num_runs = 5 if args.num_runs is None else min(int(args.num_runs), 5)

    for run_idx in range(num_runs):
        seed_eval = run_idx
        out_json = os.path.join(Config.RESULT_DIR, f"trajectory_{run_idx}.json")
        traj, score, end_step, used_plan = run(
            actions=actions,
            run_idx=run_idx,
            worker_idx=args.worker_idx,
            seed=seed_eval,
            out_path=out_json,
        )
        print(
            f"[run {run_idx}] total_reward={score:.6f}, "
            f"last_step={end_step}, saved={out_json}, traj_len={len(traj)}"
        )

    remove_workers_root()


if __name__ == "__main__":
    main()
