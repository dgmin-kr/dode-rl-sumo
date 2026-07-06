# Deep Reinforcement Learning for Dynamic Origin-Destination Matrix Estimation in Microscopic Traffic Simulations Considering Credit Assignment

<p align="center">
  <img src="readme/background_result.gif" alt="DODE-RL training animation" width="900">
</p>

This repository provides the implementation package for a reinforcement
learning-based dynamic origin-destination matrix estimation (DODE) framework
for microscopic traffic simulation calibration.

## Overview

- **What is the problem?** DODE calibrates time-dependent OD demand so that
  simulated link flows reproduce observed link-flow trajectories.

- **Why is microscopic DODE difficult?** In microscopic simulation, OD input decisions
  affect link flows through delayed, stochastic vehicle movements. This creates
  a credit assignment problem between OD inputs and downstream link-flow errors.

- **What does this project provide?** The DODE problem is formulated as a Markov
  Decision Process (MDP), and a model-free proximal policy optimization (PPO)
  agent sequentially generates OD departure decisions through direct interaction
  with SUMO.

- **What is evaluated?** The method is tested on a Nguyen-Dupuis toy network and
  a real-world highway subnetwork around Santa Clara and San Jose, California.

## Method

<p align="center">
  <img src="readme/background_method.png" alt="Method overview" width="900">
</p>

## Installation

Create a Python 3.11 environment, then install the pinned dependencies from the
repository root. Python 3.11 is required for the bundled SUMO Python bindings.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirement.txt
```

The repository includes a patched SUMO 1.27.0 runtime under `sumo_patch/`.
Experiment scripts automatically use this runtime through `sumo_patch.py`.

## Contents

| Path | Purpose |
| --- | --- |
| `1. Toy/` | Toy-network experiment, ground truth, proposed method, and baselines. |
| `2. Case study/` | Real-world case-study experiments. |
| `sumo_patch/` | Patched SUMO 1.27.0 runtime used by the experiments. |
| `sumo_patch.py` | Helper for locating and configuring the patched SUMO runtime. |
| `requirement.txt` | Python package versions. |

## Experiments

| Experiment | Network | Scale |
| --- | --- | --- |
| Toy | Nguyen-Dupuis | 4 OD pairs, 9 detectors |
| Case I | Santa Clara / San Jose | 42 OD pairs, 4 detectors |
| Case II | Santa Clara / San Jose | 42 OD pairs, 8 detectors |
| Case III | Santa Clara / San Jose | 132 OD pairs, 19 detectors |

## Training and Reproduction

Run the proposed PPO method with:

```powershell
python "<experiment>\1__RL_PPO\scripts\1__train.py"
python "<experiment>\1__RL_PPO\scripts\2__reproduce.py"
```

where `<experiment>` is one of:

```text
2. Case study\CASE 1) 42_ods_4_dets
2. Case study\CASE 2) 42_ods_8_dets
2. Case study\CASE 3) 132_ods_19_dets
```

For the toy experiment, use:

```powershell
python "1. Toy\3__RL_PPO\scripts\1__train.py"
python "1. Toy\3__RL_PPO\scripts\2__reproduce.py"
```

Training settings such as `EPOCHS`, `NUM_ENVS`, detector IDs, and OD-pair
definitions are stored in each experiment's `scripts/config.py`.

## License

Repository software is released for non-commercial research, educational, and
evaluation purposes only. Commercial use requires prior written permission from
the copyright holder. The bundled `sumo_patch/` runtime and related SUMO tooling
are third-party components with their own license terms. See `LICENSE` and
`sumo_patch/README.md`.

## Citation

If you use this code, data, or experimental results in your research, please cite:

```bibtex
@misc{min2025doderl,
  title         = {Deep Reinforcement Learning for Dynamic Origin-Destination Matrix Estimation in Microscopic Traffic Simulations Considering Credit Assignment},
  author        = {Min, Donggyu and Choi, Seongjin and Kim, Dong-Kyu},
  year          = {2025},
  eprint        = {2511.06229},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2511.06229},
  url           = {https://arxiv.org/pdf/2511.06229}
}
```
