import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from math import pi

CUSTOM_QUAD_CFG = ArticulationCfg(
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
        pos=(0.0, 0.0, 0.11),  # Higher spawn - was 0.12
        joint_pos={
            # Front Left Leg
            "base_lf1": -0.05,     # Less hip abduction (was -0.1181)
            "lf1_lf2": 0.6,        # Less knee bend (was 0.8360)
            "lf2_lf3": -1.2,       # Less ankle bend (was -1.6081)
            
            # Front Right Leg
            "base_rf1": 0.05,      # Less hip abduction (was 0.1066)
            "rf1_rf2": 0.6,        # Less knee bend (was 0.8202)
            "rf2_rf3": -1.2,       # Less ankle bend (was -1.6161)
            
            # Back Left Leg
            "base_lb1": -0.05,     # Less hip abduction (was -0.0522)
            "lb1_lb2": 0.6,        # Less knee bend (was 0.8198)
            "lb2_lb3": -1.2,       # Less ankle bend (was -1.6220)
            
            # Back Right Leg
            "base_rb1": 0.05,      # Less hip abduction (was 0.0663)
            "rb1_rb2": 0.6,        # Less knee bend (was 0.7983)
            "rb2_rb3": -1.2,       # Less ankle bend (was -1.6382)
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,

    # actuators={
    #     "leg_actuators": ImplicitActuatorCfg(
    #         joint_names_expr=[".*"],
    #         stiffness=100.0,        # Was 60.0 - much stiffer!
    #         damping=20.0,          # Was 12.0 - stronger damping
    #         effort_limit=6.0,      # Was 4.0 - more torque available
    #         velocity_limit=3.0,    # Was 2.0 - allow faster movements
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

)