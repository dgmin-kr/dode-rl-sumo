import os
import json
import shutil
import random
import multiprocessing as mp

import time
from datetime import timedelta

from pathlib import Path
from typing import Any, Callable, List, Optional

import numpy as np

os.environ.setdefault("GYM_LOG_LEVEL", "ERROR")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

try:
    import torch
except Exception:
    torch = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["axes.unicode_minus"] = False

from config import Config
from env import MySumoEnv
from run_timing import reset_run_time, write_run_time

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env

def _libsumo_worker(remote, parent_remote, env_fn_wrapper: CloudpickleWrapper) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: Optional[dict[str, Any]] = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except (EOFError, KeyboardInterrupt):
            break

class LibsumoSubprocVecEnv(SubprocVecEnv):
                                                                                            

    def __init__(self, env_fns, start_method: Optional[str] = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_libsumo_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

def _set_global_seed(seed: int):
                                                                          
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass

NEEDED_FILES = [
    "answer.csv",
    "detectors.add.xml",
    "network.net.xml",
    "run.sumocfg",
]

def clean_workers_root():
                                                                                               
    root = Config.WORKERS_ROOT
    if not os.path.exists(root):
        return
    for name in os.listdir(root):
        p = os.path.join(root, name)
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
        except Exception as e:
            print(f"[WARN] Failed to delete: {p} ({e})")

def remove_workers_root():
                                                            
    root = os.path.abspath(Config.WORKERS_ROOT)
    project_root = os.path.abspath(Config.RL_ROOT)
    if os.path.basename(root) != "workers":
        raise RuntimeError(f"Refusing to delete non-workers path: {root}")
    if os.path.commonpath([root, project_root]) != project_root:
        raise RuntimeError(f"Refusing to delete path outside experiment root: {root}")
    if os.path.isdir(root):
        shutil.rmtree(root)

def sync_worker_dir(idx: int):
       
    src = Config.BASE_WORK_DIR
    dst = Config.worker_dir(idx)
    Path(dst).mkdir(parents=True, exist_ok=True)
    for name in NEEDED_FILES:
        src_path = os.path.join(src, name)
        dst_path = os.path.join(dst, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"[missing utils file] {src_path}")
        shutil.copy2(src_path, dst_path)
    Path(os.path.join(dst, "dump")).mkdir(parents=True, exist_ok=True)

def make_env_fn(idx: int) -> Callable[[], MySumoEnv]:
    def _init():
        worker_rl_dir = Config.worker_dir(idx)
        answer_path   = Config.answer_path_for(idx)
        env = MySumoEnv(
            rl_dir=worker_rl_dir,
            sumo_binary=Config.SUMO_BINARY,
            origin_list=Config.ORIGIN_LIST,
            destination_list=Config.DESTINATION_LIST,
            input_interval=Config.INPUT_INTERVAL,
            detector_interval=Config.DETECTOR_INTERVAL,
            num_OD=Config.NUM_OD,
            num_det=Config.NUM_DET,
            state_dim=Config.STATE_DIM,
            answer_dir=answer_path,
            total_step=Config.TOTAL_STEP,
            action_unit_veh=Config.ACTION_UNIT_VEH
        )
        return env
    return _init

class EpisodeCSVAndPlotLogger(BaseCallback):
    def __init__(self, result_dir: str,
                 csv_name: str = "episode_rewards.csv",
                 png_name: str = "episode_rewards.png",
                 dpi: int = 600,
                 verbose: int = 0):
        super().__init__(verbose)
        self.result_dir = result_dir
        self.csv_path = os.path.join(result_dir, csv_name)
        self.png_path = os.path.join(result_dir, png_name)
        self.rewards: List[float] = []
        self.r_accs: List[float] = []
        self.dpi = dpi
        self.write_interval = max(1, int(Config.NUM_ENVS))
        self.last_write_episode = 0
        self.csv_rows_written = 0

        self.best_reward = -np.inf
        self.best_traj_path = os.path.join(self.result_dir, "best_trajectory.json")

        os.makedirs(self.result_dir, exist_ok=True)

        for p in (self.csv_path, self.png_path, self.best_traj_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] Failed to delete file: {p} ({e})")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        wrote_any = False

        for info in infos:
            ep = info.get("episode")
            if ep is not None and "r" in ep:
                ep_r = float(ep["r"])
                self.rewards.append(ep_r)
                self.r_accs.append(float(info.get("ep_r_acc", 0.0)))
                wrote_any = True

                traj = info.get("trajectory")
                if ep_r > self.best_reward:
                    self.best_reward = ep_r
                    if traj is not None:

                        with open(self.best_traj_path, "w", encoding="utf-8") as f:
                            for row in traj:
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if wrote_any and len(self.rewards) - self.last_write_episode >= self.write_interval:
            self._write_csv_append()
            self._write_png_overwrite()
            self.last_write_episode = len(self.rewards)
        return True

    def _on_training_end(self) -> None:
        self._write_csv_append()
        self._write_png_overwrite()

    def _write_csv_append(self):
        if not self.rewards:
            return
        if not os.path.exists(self.csv_path):
            self.csv_rows_written = 0
        if self.csv_rows_written >= len(self.rewards):
            return
        write_header = self.csv_rows_written == 0
        mode = "w" if write_header else "a"
        with open(self.csv_path, mode, encoding="utf-8") as f:
            if write_header:
                f.write("episode,reward,r_acc\n")
            for idx in range(self.csv_rows_written, len(self.rewards)):
                episode = idx + 1
                f.write(f"{episode},{self.rewards[idx]:.6f},{self.r_accs[idx]:.6f}\n")
        self.csv_rows_written = len(self.rewards)

    def _write_png_overwrite(self):
        if not self.r_accs:
            return

        x = np.arange(1, len(self.r_accs) + 1)
        y = np.asarray(self.r_accs, dtype=float)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, linewidth=1.0)
        plt.xlabel("episode")
        plt.ylabel("reward (r_acc)")
        plt.title("Episode vs. Reward")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.png_path, dpi=self.dpi, bbox_inches="tight")
        plt.close()

def build_vec_env():
    for i in range(Config.NUM_ENVS):
        sync_worker_dir(i)

    vec = LibsumoSubprocVecEnv(
        [make_env_fn(i) for i in range(Config.NUM_ENVS)],
        start_method="spawn",
    )
    vec = VecMonitor(vec)
    return vec

def run_training():
    t0 = time.perf_counter()

    Config.ensure_dirs()
    clean_workers_root()

    result_dir = Config.RESULT_DIR
    os.makedirs(result_dir, exist_ok=True)
    for name in (
        Config.CSV_NAME,
        Config.PNG_NAME,
        "best_trajectory.json",
        f"{Config.ALGORITHM.lower()}_sumo_vec.zip",
        "elapsed_time.txt",
    ):
        path = os.path.join(result_dir, name)
        if os.path.exists(path):
            os.remove(path)

    seed = 101
    _set_global_seed(seed)

    algorithm_name = Config.ALGORITHM
    hyperparams = Config.HYPERPARAMS

    vec_env = None
    try:
        vec_env = build_vec_env()
        try:
            vec_env.seed(seed)
        except Exception:
            pass

        if algorithm_name == "PPO":
            model = PPO("MlpPolicy", vec_env, seed=seed, **hyperparams["PPO"])

            with torch.no_grad():
                p = Config.INIT_ACTION_PROB
                model.policy.action_net.bias.fill_(float(np.log(p / (1.0 - p))))
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm_name}")

        ep_logger = EpisodeCSVAndPlotLogger(
            result_dir=result_dir,
            csv_name=Config.CSV_NAME,
            png_name=Config.PNG_NAME,
            dpi=Config.PLOT_DPI,
            verbose=0,
        )
        callbacks = CallbackList([ep_logger])

        total_timesteps = Config.EPOCHS * Config.TOTAL_STEP
        model.learn(total_timesteps=total_timesteps, callback=callbacks)

        save_path = os.path.join(result_dir, f"{algorithm_name.lower()}_sumo_vec")
        model.save(save_path)
    finally:
        try:
            if vec_env is not None:
                vec_env.close()
        finally:
            remove_workers_root()

    dt = time.perf_counter() - t0
    msg = f"elapsed = {timedelta(seconds=round(dt))} ({dt:.2f} sec)"
    print(msg)
    with open(os.path.join(result_dir, "elapsed_time.txt"), "w", encoding="utf-8") as f:
        f.write(msg + "\n")
    write_run_time(Config.RESULT_DIR, dt)

def main():
    Config.ensure_dirs()
    reset_run_time(Config.RESULT_DIR)
    run_training()

if __name__ == "__main__":
    main()
