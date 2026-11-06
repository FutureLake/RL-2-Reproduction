# RL² Reproduction (PyTorch + PPO)

Reproduction code for **RL²: Fast Reinforcement Learning via Slow Reinforcement Learning** ([Duan et al., 2016](https://arxiv.org/abs/1611.02779)), covering bandits, tabular MDPs and ViZDoom visual navigation. The bandit/tabular implementation builds on [Lucas Lingle's PyTorch reference](https://github.com/lucaslingle/pytorch_rl2) at a pinned commit.

This implementation uses **PPO**, rather than the original paper's TRPO. It is an independent reproduction, with explicit task protocols and reproducibility checks; it does not claim identical paper curves or exact original engine behavior. Code is provided without trained weights, datasets, experiment reports or private research material. See [attribution](NOTICE.md).

## Tasks and architecture

| Task | Protocol | Network / training |
|---|---|---|
| Bandit | 5 Bernoulli arms; 100 pulls per trial | 256-unit recurrent policy/value; PPO |
| Tabular MDP | 10 states, 5 actions; 10 episodes × 10 steps | 256-unit recurrent policy/value; PPO |
| Visual navigation | 5×5 maze; exactly 2 episodes per trial, up to 250 actions each | jointly trained CNN + action embedding + WeightNorm GRU256, shared actor–critic |

The navigation input is RGB 160×120, area-averaged to 40×30. Actions are left/right 5° or forward 0.2 cells (25.6 Doom units) over four engine ticks; this is a local physical calibration of the paper's nominal 10 cm move. Reward is `-0.04 + success - 0.001 * collision`. Entering any part of the complete goal cell succeeds; the inset red marker is nonblocking.

Hidden state and previous action/reward/done pass across the episode boundary and reset at the next trial. PPO replays complete trials through CNN and GRU, masks padding and computes GAE across episodes within a trial. Evaluation can compare retaining versus clearing hidden state in Episode 2 using the same Episode 1 prefix; this is an inference ablation, not separately trained B1.

## Repository layout

```text
src/rl2_reproduction/       algorithms, environments, collectors and evaluation
  task3/                   map generation, engine and CPU visual baseline
  task3_aligned/           WeightNorm network, 4 CPU actors + CUDA learner
  task3_scaled/            continuation, LR schedule and robust recovery
configs/                   small tests and bandit/tabular experiment recipes
scripts/                   reference checkout, local data and config generation
tests/                     numerical, environment and recovery tests
licenses/                  third-party notices
```

`third_party/`, `data/`, `runs/`, checkpoints and locally generated configurations are intentionally ignored by Git. Generate them on your own machine.

## Installation (Windows)

Python 3.11 is the supported baseline. Install Git and Conda, then:

```powershell
git clone git@github.com:FutureLake/RL-2-Reproduction.git
cd RL-2-Reproduction
conda env create -f environment.yml
conda activate rl2-reproduction
python -m pip install -e ".[navigation,dev]"
python scripts/setup_reference.py
python -m pip check
```

Microsoft MPI is required by the reference implementation even for single-process bandit/tabular training. The Conda environment includes `msmpi`; ensure its runtime DLLs are available. On Linux/macOS replace `msmpi` with Open MPI and revalidate; those platforms have not been validated here.

For CUDA, install a PyTorch build suitable for your driver using the [official selector](https://pytorch.org/get-started/locally/). CPU sampling and GPU learning are separate; installing CUDA PyTorch does not move ViZDoom sampling onto the GPU. The development GPU path was exercised with PyTorch 2.12.0+cu130; the CPU compatibility path used 2.8.0. Exact dependency resolution may need adjustment on other machines.

### Navigation compiler prerequisite

Download the Windows ACC 1.60 release from [ZDoom/acc](https://github.com/ZDoom/acc/releases/tag/1.60). Extract `acc.exe`, `zcommon.acs`, `zdefs.acs`, `zspecial.acs` and `zwvars.acs` into `third_party/acc/1.60/`. ACC compiles the small velocity-control script; motion still uses ViZDoom's collision system. No external Doom editor or pretrained encoder is required. Game textures are obtained from the installed ViZDoom/Freedoom resources and are not bundled in Git.

## Quick start: bandit and MDP

Run from the repository root; always use a new output directory.

```powershell
python -m rl2_reproduction.baseline --config configs/smoke_bandit.json --output runs/smoke_bandit
python -m rl2_reproduction.baseline --config configs/smoke_tabular_mdp.json --output runs/smoke_mdp
python -m pytest tests/test_benchmark_envs.py tests/test_reference_baseline.py -q
```

The small runs validate the training path, not convergence. Larger CUDA benchmark recipes:

```powershell
python -m rl2_reproduction.benchmark --config configs/paper_bandit_ppo.json --output runs/bandit
python -m rl2_reproduction.benchmark --config configs/paper_tabular_mdp_ppo.json --output runs/mdp
```

Inspect configs before launching: these runs have substantial budgets. Resume a benchmark with the same command plus `--resume`; use `--resume-from` only for an intentional new continuation. Outputs record config, source hashes, optimizer and RNG state, validation and held-out evaluations.

## Navigation: build data, then train

Generate all task definitions locally. The generator enumerates layouts and separates geometric symmetry groups into train/validation/test (20/3/5 groups; 1,000/200/1,000 configurations), checks reachability using real engine actions and freezes hashes. This is a finite custom corpus, not the original authors' dataset; the data build can take time.

```powershell
python scripts/build_navigation.py
python -m rl2_reproduction.task3.cli validate-env
```

A single-process CPU smoke run uses the generated raw visual dataset and a LayerNorm GRU variant:

```powershell
python -m rl2_reproduction.task3.cli train --config configs/task3/smoke.yaml --output runs/navigation_smoke
```

The main WeightNorm experiment uses stone textures and inset opaque red goal faces. Generate a configuration bound to your local data hashes, then start from random initialization:

```powershell
python scripts/configure_navigation.py --profile 5m --output local_configs/navigation_5m.yaml
python -m rl2_reproduction.task3_aligned.cli train --config local_configs/navigation_5m.yaml --output runs/navigation_5m
```

The 5M profile uses seed 0, four single-thread CPU sampling workers, a shared CUDA learner, full-trial minibatches of 8, four PPO epochs and about 50,000 actions per update. The 20M continuation loads your completed 5M checkpoint and applies warmup/hold/cosine LR scheduling:

```powershell
python scripts/configure_navigation.py --profile 20m --parent-run runs/navigation_5m --output local_configs/navigation_20m.yaml
python -m rl2_reproduction.task3_scaled.cli train --config local_configs/navigation_20m.yaml --output runs/navigation_20m
```

Budgets count complete trials; the last trial may overshoot by at most 499 actions. No automatic extension. Use the same training command plus `--resume` to restore a completed update; no implicit resume or overwrite. A stale `active.lock` requires confirming the old process has exited before removal. These profiles are intentionally strict; inspect configuration validation before changing experiment settings.

Each navigation run evaluates a fixed 30-task validation subset every five updates, all 200 validation configurations at configured milestones and the final boundary, and all 1,000 test configurations at the final checkpoint. Final evaluation includes paired retain/reset-hidden diagnostics. Task sampling is separated from evaluation RNG. Test results must not be used for hyperparameter or checkpoint selection.

## Tests and limitations

```powershell
python -m pytest -q
```

Some real-engine tests require ACC and generated scenes; CUDA tests require a suitable GPU. Numerical tests include sequence/step consistency, boundary feedback, GAE, replay probabilities, gradients, shared optimizer state, padding and checkpoint restore. Local-data-specific historical acceptance tests are not distributed.

The public layout changes package paths and binds each run to newly generated data hashes, so it cannot directly resume checkpoints from a different source layout. No checkpoint is included. A smoke run does not demonstrate memory benefits, convergence or paper-equivalent performance. Reproduce results with independent seeds and report paired sample counts, including failures and empty subsets.

## References

- [RL²: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779)
- [Lucas Lingle's reference implementation](https://github.com/lucaslingle/pytorch_rl2)
- [Yan Duan, Meta-Learning for Control](https://escholarship.org/uc/item/20z5k8dq)
- [ViZDoom](https://github.com/Farama-Foundation/ViZDoom) and [ACC](https://github.com/ZDoom/acc)
