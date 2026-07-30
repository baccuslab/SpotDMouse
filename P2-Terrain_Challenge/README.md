# Cost-Constrained Quadrupedal RL

Code, trained policies, and deployment stack for
**["Reinforcement Learning on Cost-Constrained Quadrupedal Hardware."](https://arxiv.org/abs/2607.26434)**

## What this is

A \$300 quadruped (Mini Pupper 2) walking on policies trained entirely in
simulation — and the finding that, to do it, the network **rediscovered a
solution biology already uses**.

Cheap hardware breaks the usual sim-to-real playbook. The Mini Pupper 2 runs on
a Raspberry Pi 4 + ESP32 driving **brushed \$8 position servos** with a measured
**76 ms transport delay** and no velocity or torque feedback. That delay turns
locomotion from a Markov decision process into a *partially observable* one
(a POMDP): the action you send now doesn't land for ~76 ms, so a feedforward
policy that reacts to the current observation is always reacting to stale state.
Expensive robots (\$10k–\$150k) dodge this by buying brushless quasi-direct-drive
motors with <5 ms latency and full feedback — machine fidelity in place of
algorithmic work.

Taking inspiration from biology, whose own sensorimotor loops carry
100–160 ms delays, we framed the problem as a POMDP and trained a **time-aware
(LSTM) policy** to learn a gait with action delay.

## Videos

- [Walking in sim](https://drive.google.com/file/d/1Uyh5017P83XtV9974JcwY0-OMyFR-utT/view?usp=share_link)
- [Sim and reality (combined)](https://drive.google.com/file/d/1wTMp1C6kCgLkTJz4_60SDeLPLYNQNBTI/view?usp=share_link)
- [Walking in reality](https://drive.google.com/file/d/11_WlN_X67XkxFR6eFnWp75GHwoshYI6O/view?usp=share_link)

## Repository layout

```
training/     Isaac Lab environment + PPO configs (CUDA workstation)
deployment/   On-robot controller + trained policies + learned servo models
```

### training/
Isaac Lab manager-based RL env for the Mini Pupper 2. The actuator models the
measured servo delay (`DelayedPDActuatorCfg`), which is what makes the task a
POMDP and drives the LSTM toward the CPG solution.

```
custom_quadruped/
  __init__.py          gym registration
  flat_env_cfg.py      SpotFlatEnvCfg / _PLAY  (sim 500 Hz, control 50 Hz)
  custom_quad.py       robot config file
  mdp/rewards.py       reward terms
  mdp/events.py        
  agents/rsl_rl_ppo_cfg.py   PPO runner; LSTM active, MLP/GRU/Transformer commented
                       training run (ground-truth reference)
```

Task ids: `Isaac-Velocity-Flat-Custom-Quad-v1` (train),
`...-Play-v1` (eval/export). Run with Isaac Lab's stock rsl_rl scripts:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Velocity-Flat-Custom-Quad-v1 --headless
python scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Velocity-Flat-Custom-Quad-Play-v1 --num_envs 32
```

**The LSTM is set in `agents/rsl_rl_ppo_cfg.py`** (`CustomQuadFlatPPORunnerCfg`):
`RslRlPpoActorCriticRecurrentCfg(rnn_type="lstm", rnn_hidden_dim=128, ...)`.
To reproduce another arm of the paper's comparison, swap which runner config is
uncommented (MLP / GRU / Transformer variants are in the same file).

#### Verified actuator config (matches params/env.yaml)
| param | value |
|-------|-------|
| stiffness / damping | Kp 80.0 / Kd 2.5 |
| effort_limit / velocity_limit | 0.7 / 15.0 (+ velocity_limit_sim 15.0) |
| friction / armature | 0.03 / 0.005 |
| min_delay / max_delay | 26 / 31 steps @ 500 Hz = **52–62 ms** |
| sim.dt / decimation | 0.002 (500 Hz) / 10 (50 Hz control) |

### deployment/
Runs on the robot's RPi4. Position servos report ~0 velocity/effort, so the
policy runs **open-loop**: a synthetic PD observer (or a learned per-joint MLP
servo model) reconstructs the joint-state observations the policy trained with.

```
deploy_network.py     unified LSTM/MLP controller
policies/             LSTM.pt (CPG policy), MLP.pt (MLP baseline)
servo_model/          learned per-joint MLP servo model (config + 12 joints)
  train/              trainer + playback data that produced those models
diagnostics/          IMU / observation / joint-order bring-up tools
```

```bash
python deploy_network.py lstm        # CPG policy, open-loop @ 50 Hz
python deploy_network.py lstm_25hz   # deployed preset @ 25 Hz (5-step/200 ms buffer)
python deploy_network.py mlp         # feedforward baseline
```

Requires the on-robot **MangDang Mini Pupper 2** stack
(`MangDang.mini_pupper.*`), plus torch + numpy.

#### The learned servo model
Position servos hold a commanded target and report ~0 velocity/effort, so a
policy trained with PD-actuator dynamics sees "standing still" and freezes. The
fix that made deployment work: reconstruct the missing joint-state observations
with a **learned per-joint model**

The trained models in `servo_model/` are reproducible from
`servo_model/train/`:

```bash
cd deployment/servo_model/train
python train_servo_model.py        # fits 12 per-joint MLPs from data/, writes ../servo_*.pt
```
Trained at 25 Hz, history 5 (200 ms), hidden 32 — matching `servo_model_config.json`.

---

## For previous versions
For the few who were following this repo before it became a research paper, you can revert the repo back to a prior checkpoint and fork from there to recover the archived code.
