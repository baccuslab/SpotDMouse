import gymnasium as gym
from . import agents

##
# Register Gym environments.
#
# Active envs are the "-v1" flat-terrain train/play pair used for the paper's
# results. (Earlier -v0 and rough-terrain variants were removed as unused.)
##

gym.register(
    id="Isaac-Velocity-Flat-Custom-Quad-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # <module> : <class>
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SpotFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CustomQuadFlatPPORunnerCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

# Flat-terrain PLAY (fewer envs, no randomization) — used to export policies.
gym.register(
    id="Isaac-Velocity-Flat-Custom-Quad-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SpotFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:CustomQuadFlatPPORunnerCfg"
        ),
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
