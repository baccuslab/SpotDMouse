import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Mini Pupper hybrid walking gait in Isaac Lab.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

# === Robot Config ===
cfg_robot = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd"),
    actuators={"leg_actuators": ImplicitActuatorCfg(joint_names_expr=[".*"], effort_limit_sim=100.0, velocity_limit_sim=100.0, damping=100.0, stiffness=10000.0)},
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "base_lf1": -0.1181, "lf1_lf2": 0.8360, "lf2_lf3": -1.6081,
            "base_rf1": 0.1066, "rf1_rf2": 0.8202, "rf2_rf3": -1.6161,
            "base_lb1": -0.0522, "lb1_lb2": 0.8198, "lb2_lb3": -1.6220,
            "base_rb1": 0.0663, "rb1_rb2": 0.7983, "rb2_rb3": -1.6382,
        },
        pos=(0.0, 0.0, 0.055),
    ),
)

# === Scene Config ===
class NewRobotsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    robot = cfg_robot.replace(prim_path="{ENV_REGEX_NS}/Spot")

# === Gait Controller ===
class GaitController:
    def __init__(self):
        self.step_frequency = 1.0  # Hz
        self.step_amplitude = 0.2  # rad hip swing
        self.knee_amplitude = 0.3  # rad knee lift

    def compute_joint_targets(self, time):
        phase = 2 * np.pi * self.step_frequency * time
        return {
            "base_lf1": -0.1181 + self.step_amplitude * np.sin(phase),
            "lf1_lf2": 0.8360 - self.knee_amplitude * np.maximum(0, np.sin(phase)),
            "lf2_lf3": -1.6081,
            "base_rf1": 0.1066 + self.step_amplitude * np.sin(phase + np.pi),
            "rf1_rf2": 0.8202 - self.knee_amplitude * np.maximum(0, np.sin(phase + np.pi)),
            "rf2_rf3": -1.6161,
            "base_lb1": -0.0522 + self.step_amplitude * np.sin(phase + np.pi),
            "lb1_lb2": 0.8198 - self.knee_amplitude * np.maximum(0, np.sin(phase + np.pi)),
            "lb2_lb3": -1.6220,
            "base_rb1": 0.0663 + self.step_amplitude * np.sin(phase),
            "rb1_rb2": 0.7983 - self.knee_amplitude * np.maximum(0, np.sin(phase)),
            "rb2_rb3": -1.6382,
        }

# === Main Simulation Loop ===
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    joint_name_to_index = {n: i for i, n in enumerate(scene["robot"].data.joint_names)}
    hold_duration = 1.0
    total_hold_steps = int(hold_duration / sim_dt)

    desired_standing = torch.tensor([
        -0.1181, 0.8360, -1.6081,
         0.1066, 0.8202, -1.6161,
        -0.0522, 0.8198, -1.6220,
         0.0663, 0.7983, -1.6382,
    ], device=scene["robot"].data.default_joint_pos.device).unsqueeze(0)

    print("[INFO]: Holding initial pose with high stiffness...")
    for _ in range(total_hold_steps):
        scene["robot"].set_joint_position_target(desired_standing)
        scene.write_data_to_sim()
        sim.step()

    print("[INFO]: Transitioning to walking...")
    gait_controller = GaitController()
    sim_time = 0.0

    while simulation_app.is_running():
        targets = gait_controller.compute_joint_targets(sim_time)
        joint_targets = torch.zeros_like(scene["robot"].data.default_joint_pos)

        for name, value in targets.items():
            idx = joint_name_to_index[name]
            joint_targets[:, idx] = value

        scene["robot"].set_joint_position_target(joint_targets)
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt

# === Main ===
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()


