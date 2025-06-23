import argparse
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Mini Pupper walking gait in Isaac Lab.")
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

# === Gait Controller ===
class GaitController:
    def __init__(self):
        self.contact_phases = np.array([
            [1, 0],  # front-left
            [0, 1],  # front-right
            [0, 1],  # rear-left
            [1, 0],  # rear-right
        ])
        self.phase_ticks = [50, 50]
        self.phase_length = sum(self.phase_ticks)
        self.num_phases = len(self.phase_ticks)

    def phase_index(self, ticks):
        phase_time = ticks % self.phase_length
        phase_sum = 0
        for i in range(self.num_phases):
            phase_sum += self.phase_ticks[i]
            if phase_time < phase_sum:
                return i
        assert False

    def contacts(self, ticks):
        return self.contact_phases[:, self.phase_index(ticks)]

# === Robot Config ===
cfg_robot = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd"),
    actuators={"leg_actuators": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=0.006, stiffness=10.0)},
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "base_lf1": -0.1181,
            "lf1_lf2": 0.8360,
            "lf2_lf3": -1.6081,
            "base_rf1": 0.1066,
            "rf1_rf2": 0.8202,
            "rf2_rf3": -1.6161,
            "base_lb1": -0.0522,
            "lb1_lb2": 0.8198,
            "lb2_lb3": -1.6220,
            "base_rb1": 0.0663,
            "rb1_rb2": 0.7983,
            "rb2_rb3": -1.6382,
        },
        pos=(0.0, 0.0, 0.11),
    ),
)

# === Scene Config ===
class NewRobotsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)))
    robot = cfg_robot.replace(prim_path="{ENV_REGEX_NS}/Spot")

# === Main Simulator Loop ===
def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
    
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    ticks = 0
    gait_controller = GaitController()

    joint_name_to_index = {name: idx for idx, name in enumerate(scene["robot"].data.joint_names)}
    default_joint_pos = scene["robot"].data.default_joint_pos.clone()

    leg_joint_names = {
        0: ["base_lf1", "lf1_lf2", "lf2_lf3"],
        1: ["base_rf1", "rf1_rf2", "rf2_rf3"],
        2: ["base_lb1", "lb1_lb2", "lb2_lb3"],
        3: ["base_rb1", "rb1_rb2", "rb2_rb3"],
    }

    while simulation_app.is_running():
        
        if count % 500 == 0:
            count = 0
            print("[INFO]: Resetting Mini Pupper state...")
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            scene["robot"].write_joint_state_to_sim(default_joint_pos, scene["robot"].data.default_joint_vel.clone())
            scene.reset()

        joint_pos = default_joint_pos.clone()
        contact_modes = gait_controller.contacts(ticks)

        for leg in range(4):
            if contact_modes[leg] == 0:  # Swing phase
                for joint_name in leg_joint_names[leg]:
                    idx = joint_name_to_index[joint_name]
                    if "base" in joint_name:
                        joint_pos[:, idx] += 0.05
                    elif joint_name.endswith("2"):
                        joint_pos[:, idx] -= 0.3
                    elif joint_name.endswith("3"):
                        joint_pos[:, idx] += 0.3

        scene["robot"].set_joint_position_target(joint_pos)
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        ticks += 1
        scene.update(sim_dt)

# === Main ===
def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()
