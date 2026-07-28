import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from math import pi

CUSTOM_QUAD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/mini_pupper_ros/mini_pupper_description/urdf/mini_pupper_2/mini_pupper_description/mini_pupper_description.usd",
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.10),  # Appropriate height for 45° leg angle
        # In your init_state, try slightly straighter legs:
        joint_pos={
            # 45° 'harvardrun_45'
            "base_lf1": 0.0,      
            "lf1_lf2": 0.785,     # π/4 radians = 45°
            "lf2_lf3": -1.57,     # -π/2 radians = -90° (to keep foot flat)

            "base_rf1": 0.0,      
            "rf1_rf2": 0.785,     # π/4 radians = 45°
            "rf2_rf3": -1.57,     # -π/2 radians = -90°

            "base_lb1": 0.0,      
            "lb1_lb2": 0.785,     # π/4 radians = 45°
            "lb2_lb3": -1.57,     # -π/2 radians = -90°

            "base_rb1": 0.0,      
            "rb1_rb2": 0.785,     # π/4 radians = 45°            
            "rb2_rb3": -1.57,     # -π/2 radians = -90°
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    # actuators={
    # "leg_joints": DCMotorCfg(
    #     joint_names_expr=[
    #         "base_lf1", "lf1_lf2", "lf2_lf3",  
    #         "base_rf1", "rf1_rf2", "rf2_rf3",
    #         "base_lb1", "lb1_lb2", "lb2_lb3",
    #         "base_rb1", "rb1_rb2", "rb2_rb3"
    #         ],
    #     # saturation_effort=2.5,
    #     # velocity_limit=10.0,
    #     # stiffness=45.0,        
    #     # damping=1.3,          
    #     # friction=0.02,        
    #     # armature=0.005,#0.004269, # Sweet spot - jitters in place, no drift
    #     # Accurate specs from https://www.robotshop.com/products/mangdang-high-performance-35kg-cm-robot-digital-servo?qd=cc36ca2653f9fea65ad13bd91c459f1c
    #     saturation_effort=0.35, # 3.5 kg·cm converted to N·m
    #     velocity_limit=10.5, # 0.1s/60° = 10.47 rad/s
    #     stiffness=80.0,#80.0 Official/Final: 45.0       
    #     damping=2.5,#2.0 Official/Final: 1.3     
    #     friction=0.03,        
    #     armature=0.005,#0.004269, # Sweet spot - jitters in place, no drift      
    # ),
    actuators={
    "leg_joints": DelayedPDActuatorCfg( #changed from DCMotorCfg
        joint_names_expr=[
            # LF leg (front-left)
            "base_lf1", "lf1_lf2", "lf2_lf3",
            # RF leg (front-right)  
            "base_rf1", "rf1_rf2", "rf2_rf3",
            # LB leg (back-left)
            "base_lb1", "lb1_lb2", "lb2_lb3",
            # RB leg (back-right)
            "base_rb1", "rb1_rb2", "rb2_rb3"
            ],
        # saturation_effort=0.35,
        effort_limit=0.7,
        velocity_limit=15.0,#10.5
        velocity_limit_sim=15.0,
        stiffness=80.0,#80.0
        damping=2.5,#Official:2.5       
        friction=0.03,        
        armature=0.005,#0.005
        min_delay=26,#goal:26 
        max_delay=31 ##goal:31
        #0→8/12→16/22→26/31.
    ),
    }
)


