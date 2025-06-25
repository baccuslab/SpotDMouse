# import argparse

# from isaaclab.app import AppLauncher

# # add argparse arguments
# parser = argparse.ArgumentParser(
#     description="This script demonstrates adding a custom robot to an Isaac Lab environment."
# )
# parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app

# import numpy as np
# import torch

# import isaaclab.sim as sim_utils
# from isaaclab.actuators import ImplicitActuatorCfg
# from isaaclab.assets import AssetBaseCfg
# from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# import isaaclab.sim as sim_utils
# from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
# from isaaclab.assets.articulation import ArticulationCfg
# import os
# from math import pi

# # cfg_robot = ArticulationCfg(
# #     spawn=sim_utils.UsdFileCfg(usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd"),
# #     actuators={"leg_actuators": ImplicitActuatorCfg(joint_names_expr=[".*"],damping=1.0,stiffness=0.0)},
# # )

# # cfg_robot = ArticulationCfg(
# #     spawn=sim_utils.UsdFileCfg(usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd"),
# #     actuators={"leg_actuators": ImplicitActuatorCfg(joint_names_expr=[".*"], damping=0.006, stiffness=10.0)},
# #     init_state=ArticulationCfg.InitialStateCfg(
# #         joint_pos={
# #             "base_lf1": -0.1181,
# #             "lf1_lf2": 0.8360,
# #             "lf2_lf3": -1.6081,
# #             "base_rf1": 0.1066,
# #             "rf1_rf2": 0.8202,
# #             "rf2_rf3": -1.6161,
# #             "base_lb1": -0.0522,
# #             "lb1_lb2": 0.8198,
# #             "lb2_lb3": -1.6220,
# #             "base_rb1": 0.0663,
# #             "rb1_rb2": 0.7983,
# #             "rb2_rb3": -1.6382,
# #         },
# #         pos=(0.0, 0.0, 0.11),  # This base height is fine to prevent collisions on load
# #     ),
# # )

# cfg_robot = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             rigid_body_enabled=True,
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.1,
#             angular_damping=0.1,
#             max_linear_velocity=1.0,
#             max_angular_velocity=3.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=8,
#             solver_velocity_iteration_count=1,
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.11),
#         joint_pos={
#             "base_lf1": -0.1181,
#             "lf1_lf2": 0.8360,
#             "lf2_lf3": -1.6081,
#             "base_rf1": 0.1066,
#             "rf1_rf2": 0.8202,
#             "rf2_rf3": -1.6161,
#             "base_lb1": -0.0522,
#             "lb1_lb2": 0.8198,
#             "lb2_lb3": -1.6220,
#             "base_rb1": 0.0663,
#             "rb1_rb2": 0.7983,
#             "rb2_rb3": -1.6382,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "leg_actuators": DCMotorCfg(
#             joint_names_expr=[".*"],
#             effort_limit=0.6,       # 4x higher effort
#             saturation_effort=0.6,
#             velocity_limit=0.2,     # Slightly faster corrections
#             stiffness=1.6,          # 4x higher stiffness
#             damping=1.6,            # Critical damping matches stiffness
#             friction=0.1,
#         )
#     },
# )


# class NewRobotsSceneCfg(InteractiveSceneCfg):
#     """Designs the scene."""

#     # Ground-plane
#     ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

#     # lights
#     dome_light = AssetBaseCfg(
#         prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
#     )

#     # robot
#     robot = cfg_robot.replace(prim_path="{ENV_REGEX_NS}/Spot")

# def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
#     print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
#     sim_dt = sim.get_physics_dt()
#     sim_time = 0.0
#     count = 0

#     # Cache standing joint position target
#     default_joint_pos = scene["robot"].data.default_joint_pos.clone()

#     while simulation_app.is_running():
#         # Periodic reset
#         if count % 500 == 0:
#             count = 0
#             print("[INFO]: Resetting Mini Pupper state...")

#             # Reset root pose and velocity
#             root_state = scene["robot"].data.default_root_state.clone()
#             root_state[:, :3] += scene.env_origins
#             scene["robot"].write_root_pose_to_sim(root_state[:, :7])
#             scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])

#             # Reset joints
#             scene["robot"].write_joint_state_to_sim(default_joint_pos, scene["robot"].data.default_joint_vel.clone())

#             # Clear internal buffers
#             scene.reset()

#         # Continuously apply standing joint targets to hold position
#         scene["robot"].set_joint_position_target(default_joint_pos)

#         # Step sim
#         scene.write_data_to_sim()
#         sim.step()
#         sim_time += sim_dt
#         count += 1
#         scene.update(sim_dt)


# def main():
#     """Main function."""
#     # Initialize the simulation context
#     sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
#     sim = sim_utils.SimulationContext(sim_cfg)

#     sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
#     # design scene
#     scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
#     scene = InteractiveScene(scene_cfg)
#     # Play the simulator
#     sim.reset()
#     # Now we are ready!
#     print("[INFO]: Setup complete...")
#     # Run the simulator
#     run_simulator(sim, scene)


# if __name__ == "__main__":
#     main()
#     simulation_app.close()
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg, DCMotorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import os
from math import pi

# ACCURATE for real MiniPupper: 560g total weight
cfg_robot = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/mini_pupper_ros/mini_pupper_description/usd/generated_mini_pupper/generated_mini_pupper.usd",
        activate_contact_sensors=True,
        # Realistic mass distribution for 560g MiniPupper
        mass_props=sim_utils.MassPropertiesCfg(
            mass=0.45,  # ~450g in base (80% of total weight: battery, Pi, PCBs)
                       # Leaves ~110g for all 12 leg segments (carbon fiber)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.005,   # Very low for small, lightweight robot
            angular_damping=0.005,  # Very low - carbon fiber has minimal drag
            max_linear_velocity=15.0,  # Small robots can be quite fast
            max_angular_velocity=25.0, # High agility for lightweight design
            max_depenetration_velocity=2.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=6,   # Lower for lightweight robot
            solver_velocity_iteration_count=1,   # Standard
            sleep_threshold=0.01,               # Appropriate for 560g robot
            stabilization_threshold=0.002,      # Good balance
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.09),  # Conservative standing height
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
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    # Tuned for MiniPupper's actual servo specs (likely SG90/MG90S micro servos)
    # More realistic for RL if you want servo dynamics
    # actuators={
    #     "leg_actuators": DCMotorCfg(
    #         joint_names_expr=[".*"],
    #         saturation_effort=3.0,
    #         velocity_limit=1.5,
    #         stiffness=50.0,        # Lower for more realistic servo response
    #         damping=10.0,          # Lower for more realistic dynamics
    #         friction=0.05,        # Servo friction
    #         armature=0.001,       # Small servo inertia
    #     )
    # }
    actuators={
        "leg_actuators": DCMotorCfg(
            joint_names_expr=[".*"],
            saturation_effort=5.0,     # High torque
            velocity_limit=3.0,
            stiffness=80.0,           # Very stiff
            damping=16.0,             # Strong damping
            friction=0.01,            # Low friction
            armature=0.001,
        )
    }    
    # Alternative: If using higher-torque servos or want to match URDF
    # actuators={
    #     "leg_actuators": ImplicitActuatorCfg(
    #         joint_names_expr=[".*"],
    #         stiffness=25.0,    
    #         damping=5.0,      
    #         effort_limit=3.0,  # Higher if using better servos
    #     )
    # },
)

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""
    
    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    
    # robot
    robot = cfg_robot.replace(prim_path="{ENV_REGEX_NS}/Spot")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    print("[DEBUG]: Joint Names:", scene["robot"].data.joint_names)
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    
    # Cache standing joint position target
    default_joint_pos = scene["robot"].data.default_joint_pos.clone()
    
    # Add small settling period after spawn
    settling_steps = 100
    
    while simulation_app.is_running():
        # Periodic reset
        if count % 1000 == 0:  # Increased reset interval
            count = 0
            print("[INFO]: Resetting Mini Pupper state...")
            
            # Reset root pose and velocity
            root_state = scene["robot"].data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            # Add small random perturbation to test stability
            root_state[:, 2] += torch.randn_like(root_state[:, 2]) * 0.01
            
            scene["robot"].write_root_pose_to_sim(root_state[:, :7])
            scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
            
            # Reset joints with small random perturbation
            joint_pos_reset = default_joint_pos.clone()
            joint_pos_reset += torch.randn_like(joint_pos_reset) * 0.05
            scene["robot"].write_joint_state_to_sim(joint_pos_reset, scene["robot"].data.default_joint_vel.clone())
            
            # Clear internal buffers
            scene.reset()
        
        # Apply standing joint targets with gentle settling
        if count < settling_steps:
            # Gradual settling - interpolate to target position
            alpha = count / settling_steps
            current_pos = scene["robot"].data.joint_pos
            target_pos = alpha * default_joint_pos + (1 - alpha) * current_pos
            scene["robot"].set_joint_position_target(target_pos)
        else:
            # Normal position control
            scene["robot"].set_joint_position_target(default_joint_pos)
        
        # Step sim
        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)
        
        # Debug output every 100 steps
        if count % 100 == 0:
            root_pos = scene["robot"].data.root_pos_w[0]
            root_quat = scene["robot"].data.root_quat_w[0]
            print(f"[DEBUG] Step {count}: Root pos: {root_pos}, Root quat: {root_quat}")

def main():
    """Main function."""
    # Simulation tuned for 2 lb robot with carbon fiber legs
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device,
        dt=0.004,  # 250Hz - good balance for lightweight robot
        physx=sim_utils.PhysxCfg(
            solver_type=1,  # TGS solver
            enable_stabilization=True,
            bounce_threshold_velocity=0.15,  # Appropriate for lightweight
            friction_offset_threshold=0.03,
            friction_correlation_distance=0.02,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.3])
    
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Play the simulator
    sim.reset()
    
    # Now we are ready!
    print("[INFO]: Setup complete...")
    print("[INFO]: Robot should spawn and settle into stable standing position")
    
    # Run the simulator
    run_simulator(sim, scene)

if __name__ == "__main__":
    main()
    simulation_app.close()