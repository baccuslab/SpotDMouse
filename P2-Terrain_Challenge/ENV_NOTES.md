# Environment / Reproducibility Notes

## Training workstation
- GPU: NVIDIA RTX 4090 (24 GiB)
- Isaac Sim version:  `4.5.0`
- Isaac Lab version:  `2.1.0`
- PyTorch / CUDA:     `2.0.1+cu117`
- Parallel envs used: 4096

## Training-side Python deps
Isaac Lab bundles its own environment; the task package only needs:
- gymnasium
- torch
- (Isaac Lab + Isaac Sim, installed per NVIDIA's instructions)

## Robot (Raspberry Pi 4) deps
- MangDang Mini Pupper 2 software stack (provides `MangDang.mini_pupper.*`)
  — pre-installed on the robot image; not pip-installable standalone.
- torch (CPU)
- numpy
