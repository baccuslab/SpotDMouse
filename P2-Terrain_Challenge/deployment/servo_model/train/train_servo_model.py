import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# ─── Config ──────────────────────────────────────────────────────────
HISTORY = 5          # cmd steps [t, t-1, ..., t-4] → 200ms at 25Hz
INPUT_DIM = HISTORY + 1  # 5 cmd_disp + 1 real_disp = 6
HIDDEN = 32
EPOCHS = 2000
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64

SIM_DEFAULTS = np.array([0.0, 0.785, -1.57] * 4)
JOINT_NAMES = [
    "LF_hip", "LF_thigh", "LF_calf",
    "RF_hip", "RF_thigh", "RF_calf",
    "LB_hip", "LB_thigh", "LB_calf",
    "RB_hip", "RB_thigh", "RB_calf",
]

# Repo-relative paths. This script lives in servo_model/train/; it writes the
# trained models up one level into servo_model/ (where deploy_network.py loads
# them from) and reads its playback data from servo_model/train/data/.
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUT_DIR = SCRIPT_DIR.parent            # -> deployment/servo_model/
OUT_DIR.mkdir(exist_ok=True)

# ─── Load data ───────────────────────────────────────────────────────
print("Loading playback data...")
pb = pd.read_csv(DATA_DIR / "playback_lstm_25hz.csv")

cmd_pos = np.column_stack([pb[f"cmd_pos_{i}"].values for i in range(12)])
real_pos = np.column_stack([pb[f"real_pos_{i}"].values for i in range(12)])
loops = pb["loop"].values

cmd_disp = cmd_pos - SIM_DEFAULTS[None, :]
real_disp = real_pos - SIM_DEFAULTS[None, :]

print(f"  {len(pb)} samples, {len(np.unique(loops))} loops")
print(f"  cmd_disp range: [{cmd_disp.min():.3f}, {cmd_disp.max():.3f}]")
print(f"  real_disp range: [{real_disp.min():.3f}, {real_disp.max():.3f}]")


# ─── Build windowed dataset per joint ────────────────────────────────
def build_dataset(cmd_d, real_d, loop_ids, history=HISTORY):
    """Build (X, y) pairs, respecting loop boundaries."""
    X_all, y_all, loop_labels = [], [], []

    for loop_id in np.unique(loop_ids):
        mask = loop_ids == loop_id
        c = cmd_d[mask]   # (T, 12)
        r = real_d[mask]   # (T, 12)
        T = len(c)

        for t in range(history, T - 1):
            # Input: cmd_disp[t], cmd_disp[t-1], ..., cmd_disp[t-history+1], real_disp[t]
            # Shape per joint: (history + 1,)
            cmd_window = c[t - history + 1 : t + 1][::-1]  # [t, t-1, ..., t-K+1]
            x_j = np.concatenate([cmd_window, r[t:t+1]], axis=0)  # (history+1, 12)
            X_all.append(x_j)
            y_all.append(r[t + 1])
            loop_labels.append(loop_id)

    X = np.array(X_all)  # (N, history+1, 12)
    y = np.array(y_all)  # (N, 12)
    loop_labels = np.array(loop_labels)
    return X, y, loop_labels


print("\nBuilding windowed dataset...")
X, y, loop_labels = build_dataset(cmd_disp, real_disp, loops)
print(f"  Dataset: {X.shape[0]} samples, input shape per joint: ({INPUT_DIM},)")

# Train/val split: loops 0+1 for train, loop 2 for val
train_mask = loop_labels <= 1
val_mask = loop_labels == 2

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
print(f"  Train: {len(X_train)}, Val: {len(X_val)}")


# ─── Per-joint MLP ───────────────────────────────────────────────────
class JointServoMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden=HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ─── Training ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING PER-JOINT SERVO MODELS")
print("=" * 60)

models = {}
train_losses = {}
val_losses = {}

for j in range(12):
    name = JOINT_NAMES[j]

    # Extract per-joint data: X is (N, history+1, 12) → take joint j
    xtr = torch.tensor(X_train[:, :, j], dtype=torch.float32)
    ytr = torch.tensor(y_train[:, j], dtype=torch.float32)
    xva = torch.tensor(X_val[:, :, j], dtype=torch.float32)
    yva = torch.tensor(y_val[:, j], dtype=torch.float32)

    model = JointServoMLP()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val = float("inf")
    best_state = None
    t_losses, v_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        # Shuffle
        idx = torch.randperm(len(xtr))
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(xtr), BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]
            xb, yb = xtr[batch_idx], ytr[batch_idx]
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(xva)
            val_loss = nn.functional.mse_loss(val_pred, yva).item()

        t_losses.append(epoch_loss)
        v_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    models[name] = model
    train_losses[name] = t_losses
    val_losses[name] = v_losses

    # Stats
    model.eval()
    with torch.no_grad():
        tr_pred = model(xtr).numpy()
        va_pred = model(xva).numpy()

    tr_rmse = np.sqrt(np.mean((tr_pred - ytr.numpy()) ** 2))
    va_rmse = np.sqrt(np.mean((va_pred - yva.numpy()) ** 2))

    # Baseline: predict real_disp[t+1] = real_disp[t] (persistence)
    persist_tr = xtr[:, -1].numpy()  # last element is real_disp[t]
    persist_va = xva[:, -1].numpy()
    persist_tr_rmse = np.sqrt(np.mean((persist_tr - ytr.numpy()) ** 2))
    persist_va_rmse = np.sqrt(np.mean((persist_va - yva.numpy()) ** 2))

    print(f"  {name:<12} train RMSE={tr_rmse:.5f}  val RMSE={va_rmse:.5f}  "
          f"(baseline: tr={persist_tr_rmse:.5f} va={persist_va_rmse:.5f})  "
          f"improvement: {(1 - va_rmse/persist_va_rmse)*100:+.1f}%")


# ─── Autoregressive rollout evaluation ───────────────────────────────
# This is the real test: feed predictions back in, like deployment
print("\n" + "=" * 60)
print("AUTOREGRESSIVE ROLLOUT (deployment-realistic)")
print("=" * 60)

def autoregressive_rollout(models, cmd_disp_seq, real_disp_init, history=HISTORY):
    """
    Roll out servo models autoregressively.
    cmd_disp_seq: (T, 12) — the commanded displacements
    real_disp_init: (12,) — initial real displacement (step 0)
    Returns: predicted real_disp (T, 12)
    """
    T = len(cmd_disp_seq)
    pred = np.zeros((T, 12))
    pred[0] = real_disp_init.copy()

    # Pad cmd history with zeros for the first few steps
    cmd_padded = np.vstack([np.zeros((history - 1, 12)), cmd_disp_seq])

    for t in range(T - 1):
        for j, name in enumerate(JOINT_NAMES):
            # Build input: cmd_disp[t, t-1, ..., t-K+1], real_disp_pred[t]
            cmd_window = cmd_padded[t + history - 1 : t - 1 + history - 1 : -1, j] if t > 0 else cmd_padded[history - 1 : : -1, j]
            # More careful indexing
            cmd_indices = [t + (history - 1) - k for k in range(history)]
            cmd_window = cmd_padded[cmd_indices, j]

            x = np.concatenate([cmd_window, [pred[t, j]]])
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                pred[t + 1, j] = models[name](x_t).item()

    return pred


# Rollout on each loop
fig, axes = plt.subplots(4, 3, figsize=(22, 20), sharex=True)
fig.suptitle("Autoregressive Servo Model Rollout vs Actual\n"
             "Green=Cmd disp, Red=Actual real, Blue=Model prediction (autoregressive)",
             fontsize=14, fontweight="bold")

rollout_rmses = {n: [] for n in JOINT_NAMES}

for loop_id in range(3):
    mask = loops == loop_id
    c = cmd_disp[mask]
    r = real_disp[mask]

    pred = autoregressive_rollout(models, c, r[0])

    for j in range(12):
        ax = axes[j // 3, j % 3]
        T = len(c)
        t = np.arange(T) * 0.04  # 25Hz → 40ms

        if loop_id == 0:
            ax.plot(t, r[:, j], "r-", lw=1.3, alpha=0.8, label="Actual real")
            ax.plot(t, c[:, j], "g--", lw=0.8, alpha=0.5, label="Cmd disp")
            ax.plot(t, pred[:, j], "b-", lw=1.3, alpha=0.8, label="Model pred")
        else:
            ax.plot(t + loop_id * T * 0.04, r[:, j], "r-", lw=1.3, alpha=0.8)
            ax.plot(t + loop_id * T * 0.04, c[:, j], "g--", lw=0.8, alpha=0.5)
            ax.plot(t + loop_id * T * 0.04, pred[:, j], "b-", lw=1.3, alpha=0.8)

        rmse = np.sqrt(np.mean((pred[:, j] - r[:, j]) ** 2))
        rollout_rmses[JOINT_NAMES[j]].append(rmse)

for j in range(12):
    ax = axes[j // 3, j % 3]
    mean_rmse = np.mean(rollout_rmses[JOINT_NAMES[j]])
    ax.set_title(f"{JOINT_NAMES[j]}  RMSE={mean_rmse:.4f} rad", fontsize=10, fontweight="bold")
    ax.set_ylabel("Displacement (rad)")
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8)

    # Add loop boundary lines
    for li in range(1, 3):
        T_loop = 150 if li == 1 else 120
        ax.axvline(li * 150 * 0.04 if li == 1 else 150 * 0.04 + 120 * 0.04,
                   color="gray", lw=0.5, ls=":", alpha=0.5)

for i in range(3):
    axes[-1, i].set_xlabel("Time (s)")

plt.tight_layout()
fig.savefig(OUT_DIR / "01_autoregressive_rollout.png", dpi=150)
plt.close()

print("\n  Autoregressive rollout RMSE (rad):")
print(f"  {'Joint':<12} {'Loop 0':>10} {'Loop 1':>10} {'Loop 2':>10} {'Mean':>10}")
for name in JOINT_NAMES:
    rs = rollout_rmses[name]
    print(f"  {name:<12} {rs[0]:10.4f} {rs[1]:10.4f} {rs[2]:10.4f} {np.mean(rs):10.4f}")


# ─── Compare: model rollout vs PD sim vs actual ─────────────────────
# Load sim data to add PD sim comparison
print("\n=== Comparison: Servo Model vs PD Sim vs Actual ===")

sim_obs = pd.read_csv(DATA_DIR / "env_2_observations.csv")
SIM_COL_NAMES = [
    "base_lf1", "lf1_lf2", "lf2_lf3",
    "base_rf1", "rf1_rf2", "rf2_rf3",
    "base_lb1", "lb1_lb2", "lb2_lb3",
    "base_rb1", "rb1_rb2", "rb2_rb3",
]
sim_joint_pos = np.column_stack([sim_obs[f"joint_pos_{n}"].values for n in SIM_COL_NAMES])
sim_disp = sim_joint_pos - SIM_DEFAULTS[None, :]

# Rollout on loop 0 (same actions as sim)
mask0 = loops == 0
c0 = cmd_disp[mask0]
r0 = real_disp[mask0]
pred0 = autoregressive_rollout(models, c0, r0[0])

N_cmp = min(len(sim_disp), len(c0))
sim_time = np.arange(N_cmp) / 25.0

fig, axes = plt.subplots(4, 3, figsize=(22, 20), sharex=True)
fig.suptitle("Three-Way Comparison: Sim PD vs Servo Model vs Actual Real\n"
             "Blue=Sim PD, Orange=Servo Model (autoregressive), Red=Actual Real, Green=Cmd (dashed)",
             fontsize=13, fontweight="bold")

for j in range(12):
    ax = axes[j // 3, j % 3]
    N = N_cmp

    ax.plot(sim_time[:N], sim_disp[:N, j], "b-", lw=1.3, alpha=0.8, label="Sim PD")
    ax.plot(sim_time[:N], pred0[:N, j], "-", color="darkorange", lw=1.5, alpha=0.9,
            label="Servo model")
    ax.plot(sim_time[:N], r0[:N, j], "r-", lw=1.2, alpha=0.7, label="Actual real")
    ax.plot(sim_time[:N], c0[:N, j], "g--", lw=0.8, alpha=0.4, label="Cmd disp")

    rms_pd = np.sqrt(np.mean((sim_disp[:N, j] - r0[:N, j]) ** 2))
    rms_model = np.sqrt(np.mean((pred0[:N, j] - r0[:N, j]) ** 2))

    ax.set_title(f"{JOINT_NAMES[j]}  PD gap={rms_pd:.3f}  Model gap={rms_model:.4f}",
                 fontsize=9, fontweight="bold")
    ax.set_ylabel("Displacement (rad)")
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8)

for i in range(3):
    axes[-1, i].set_xlabel("Time (s)")

plt.tight_layout()
fig.savefig(OUT_DIR / "02_three_way_comparison.png", dpi=150)
plt.close()

# Print gap comparison
print(f"\n  {'Joint':<12} {'PD→Real':>10} {'Model→Real':>10} {'Reduction':>10}")
for j, name in enumerate(JOINT_NAMES):
    rms_pd = np.sqrt(np.mean((sim_disp[:N_cmp, j] - r0[:N_cmp, j]) ** 2))
    rms_model = np.sqrt(np.mean((pred0[:N_cmp, j] - r0[:N_cmp, j]) ** 2))
    pct = (1 - rms_model / rms_pd) * 100 if rms_pd > 1e-6 else 0
    print(f"  {name:<12} {rms_pd:10.4f} {rms_model:10.4f} {pct:+10.1f}%")


# ─── Normalized waveform comparison ──────────────────────────────────
print("\n=== Normalized Waveform: Servo Model vs Actual ===")

def normalize_wave(x, skip=10):
    mu = np.mean(x[skip:])
    sigma = np.std(x[skip:])
    if sigma < 1e-6:
        return x - mu
    return (x - mu) / sigma

fig, axes = plt.subplots(4, 3, figsize=(22, 20), sharex=True)
fig.suptitle("Normalized Waveform: Does the Servo Model Capture the Wave Shape?\n"
             "Blue=Sim, Orange=Servo Model, Red=Actual Real | Pearson r for shape match",
             fontsize=13, fontweight="bold")

for j in range(12):
    ax = axes[j // 3, j % 3]
    N = N_cmp

    s_n = normalize_wave(sim_disp[:N, j])
    m_n = normalize_wave(pred0[:N, j])
    r_n = normalize_wave(r0[:N, j])

    ax.plot(sim_time[:N], s_n, "b-", lw=1.3, alpha=0.7, label="Sim")
    ax.plot(sim_time[:N], m_n, "-", color="darkorange", lw=1.5, alpha=0.9, label="Servo model")
    ax.plot(sim_time[:N], r_n, "r-", lw=1.2, alpha=0.7, label="Actual")

    SKIP = 10
    r_model_actual = np.corrcoef(m_n[SKIP:], r_n[SKIP:])[0, 1] if np.std(r_n[SKIP:]) > 1e-6 and np.std(m_n[SKIP:]) > 1e-6 else 0
    r_sim_actual = np.corrcoef(s_n[SKIP:], r_n[SKIP:])[0, 1] if np.std(r_n[SKIP:]) > 1e-6 and np.std(s_n[SKIP:]) > 1e-6 else 0
    r_model_sim = np.corrcoef(m_n[SKIP:], s_n[SKIP:])[0, 1] if np.std(s_n[SKIP:]) > 1e-6 and np.std(m_n[SKIP:]) > 1e-6 else 0

    ax.set_title(f"{JOINT_NAMES[j]}  model↔real={r_model_actual:.2f}  "
                 f"sim↔real={r_sim_actual:.2f}  model↔sim={r_model_sim:.2f}",
                 fontsize=8, fontweight="bold")
    ax.set_ylabel("Normalized")
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8)

for i in range(3):
    axes[-1, i].set_xlabel("Time (s)")

plt.tight_layout()
fig.savefig(OUT_DIR / "03_normalized_waveform.png", dpi=150)
plt.close()


# ─── Training curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(4, 3, figsize=(18, 16))
fig.suptitle("Training Curves (MSE Loss)", fontsize=14, fontweight="bold")

for j in range(12):
    ax = axes[j // 3, j % 3]
    name = JOINT_NAMES[j]
    ax.semilogy(train_losses[name], "b-", lw=0.8, alpha=0.7, label="Train")
    ax.semilogy(val_losses[name], "r-", lw=0.8, alpha=0.7, label="Val")
    ax.set_title(name, fontsize=11, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)
    if j == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
fig.savefig(OUT_DIR / "04_training_curves.png", dpi=150)
plt.close()


# ─── Save models ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

# Save as TorchScript for deployment
for name, model in models.items():
    model.eval()
    example = torch.randn(1, INPUT_DIM)
    scripted = torch.jit.trace(model, example)
    path = OUT_DIR / f"servo_{name}.pt"
    scripted.save(str(path))
    print(f"  Saved: {path}")

# Save combined numpy weights for lightweight deployment (no torch needed)
weights = {}
for name, model in models.items():
    w = {}
    for pname, param in model.named_parameters():
        w[pname] = param.detach().numpy().tolist()
    weights[name] = w

import json
with open(OUT_DIR / "servo_model_weights.json", "w") as f:
    json.dump(weights, f)
print(f"  Saved: {OUT_DIR / 'servo_model_weights.json'}")

# Save model config
config = {
    "history": HISTORY,
    "input_dim": INPUT_DIM,
    "hidden": HIDDEN,
    "sim_defaults": SIM_DEFAULTS.tolist(),
    "joint_names": JOINT_NAMES,
    "trained_at_hz": 25,
    "hw_scale_trained": 0.55,
    "description": "Per-joint servo model: real_disp[t+1] = f(cmd_disp[t:t-K], real_disp[t])",
}
with open(OUT_DIR / "servo_model_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"  Saved: {OUT_DIR / 'servo_model_config.json'}")


# ─── Print deployment integration snippet ────────────────────────────
print("\n" + "=" * 60)
print("DEPLOYMENT INTEGRATION")
print("=" * 60)
print("""
Replace the PD sim in deploy_network.py with:

class ServoModel:
    def __init__(self, model_dir, history=5):
        self.history = history
        self.models = {}
        JOINT_NAMES = [
            "LF_hip", "LF_thigh", "LF_calf",
            "RF_hip", "RF_thigh", "RF_calf",
            "LB_hip", "LB_thigh", "LB_calf",
            "RB_hip", "RB_thigh", "RB_calf",
        ]
        for name in JOINT_NAMES:
            self.models[name] = torch.jit.load(f"{model_dir}/servo_{name}.pt")
            self.models[name].eval()

        self.sim_defaults = np.array([0.0, 0.785, -1.57] * 4)
        self.cmd_buf = deque(maxlen=history)
        for _ in range(history):
            self.cmd_buf.append(np.zeros(12))
        self.pred_disp = np.zeros(12)

    def step(self, cmd_pos):
        cmd_disp = cmd_pos - self.sim_defaults
        self.cmd_buf.append(cmd_disp.copy())

        for j, name in enumerate(JOINT_NAMES):
            cmd_window = np.array([self.cmd_buf[-(k+1)][j]
                                   for k in range(self.history)])
            x = np.append(cmd_window, self.pred_disp[j])
            x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                self.pred_disp[j] = self.models[name](x_t).item()

        # Return what deploy_network needs for obs
        self.syn_pos_rel = self.pred_disp.copy()
        # Estimate velocity from position delta (finite difference)
        # This is approximate but matches what real servos would produce
        self.syn_vel = np.zeros(12)  # or finite diff if needed

    def reset(self):
        for _ in range(self.history):
            self.cmd_buf.append(np.zeros(12))
        self.pred_disp = np.zeros(12)
        self.syn_pos_rel = np.zeros(12)
        self.syn_vel = np.zeros(12)
""")

print("=" * 60)
print(f"ALL OUTPUTS IN: {OUT_DIR}")
print("=" * 60)
