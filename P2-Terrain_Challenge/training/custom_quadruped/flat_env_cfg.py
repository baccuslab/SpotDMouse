

import torch

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from dataclasses import fields
import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# from isaaclab_assets.robots.spot import SPOT_CFG  # isort: skip
# Gotta change this simple change
from isaaclab_tasks.manager_based.locomotion.velocity.config.custom_quadruped_2.custom_quad import CUSTOM_QUAD_CFG 


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.05,#OG: 0.1
    vertical_scale=0.003,#OG: 0.005
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.50, 
            noise_range=(0.01, 0.02),# OG:(0.02, 0.05)
            noise_step=0.01, #OG: 0.02 
            border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.15),    # gentle — max ~8.5°
            platform_width=0.6,          # scaled to Mini Pupper size
            border_width=0.25,
            inverted=False,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15,
            slope_range=(0.0, 0.15),
            platform_width=0.6,
            border_width=0.25,
            inverted=True,
        ),
    },
)


@configclass
class SpotActionsCfg:
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[
            # LF leg (front-left)
            "base_lf1", "lf1_lf2", "lf2_lf3",
            # RF leg (front-right)  
            "base_rf1", "rf1_rf2", "rf2_rf3",
            # LB leg (back-left)
            "base_lb1", "lb1_lb2", "lb2_lb3",
            # RB leg (back-right)
            "base_rb1", "rb1_rb2", "rb2_rb3"
        ], 
        scale=0.5,#0.2 original for spot servos
        use_default_offset=True,
        preserve_order=True
    )


@configclass
class SpotCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),##first_train:2025-08-07_18-42-54=(4.0, 8.0)(next) second run resampling_time_range=(10.0, 10.0)
        rel_standing_envs=0.25,# curreculum begins with 0.05
        rel_heading_envs=0.0,
        heading_command=False, #heading_command=False
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # lin_vel_x=(-0.35, 0.35), 
            # lin_vel_y=(-0.35, 0.35), 
            # ang_vel_z=(-0.30, 0.30),
            # heading = (-3.14, 3.14)
            lin_vel_x=(0.15, 0.15),    
            lin_vel_y=(0.0, 0.0),    
            ang_vel_z=(0.0, 0.0),
            # heading=(0.0, 0.0)
        ),
    )

@configclass
class SpotCommandsCfg_PLAY:
    """Fixed command specifications for the MDP in play mode."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e10, 1.0e10), # Set a very long time so it doesn't resample
        rel_standing_envs=0.0, # Disable standing-only environments
        rel_heading_envs=0.0,  # Disable heading-only environments
        heading_command=False,#heading_command=False
        debug_vis=True,
        # SET YOUR DESIRED FIXED COMMAND VALUES HERE
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # lin_vel_x=(-0.35, 0.35), 
            # lin_vel_y=(-0.35, 0.35), 
            # ang_vel_z=(-0.30, 0.30),
            # heading=(0.0, 0.0)
            lin_vel_x=(0.15, 0.15),    
            lin_vel_y=(0.0, 0.0),    
            ang_vel_z=(0.0, 0.0),       
        ),
    )

@configclass
class SpotObservationsCfg:
    """Observation specifications for the MDP."""
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        # Base velocities (3 dims)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.2, n_max=0.2) #n_min=-0.2, n_max=0.2
        )
        
        # Orientation (3 dims for projected gravity)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05), #prev: -0.05, n_max=0.05)
        )
        
        # Commands (3 dims)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, 
            params={"command_name": "base_velocity"}
        )
        
        # Joint states (36 dims)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "base_lf1", "lf1_lf2", "lf2_lf3",
                    "base_rf1", "rf1_rf2", "rf2_rf3",
                    "base_lb1", "lb1_lb2", "lb2_lb3",
                    "base_rb1", "rb1_rb2", "rb2_rb3"
                ])
            }, 
            noise=Unoise(n_min=-0.01, n_max=0.01)#prev: n_min=-0.01, n_max=0.01 tes: n_min=-0.05 n_max=0.05
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "base_lf1", "lf1_lf2", "lf2_lf3",
                    "base_rf1", "rf1_rf2", "rf2_rf3",
                    "base_lb1", "lb1_lb2", "lb2_lb3",
                    "base_rb1", "rb1_rb2", "rb2_rb3"
                ])
            }, 
            noise=Unoise(n_min=-1.5, n_max=1.5) #prev: n_min=-1.5, n_max=1.5
        )

        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[
                    "base_lf1", "lf1_lf2", "lf2_lf3",
                    "base_rf1", "rf1_rf2", "rf2_rf3",
                    "base_lb1", "lb1_lb2", "lb2_lb3",
                    "base_rb1", "rb1_rb2", "rb2_rb3"
                ])
            },
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        
        # Previous actions (12 dims)
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True #Was False
            self.concatenate_terms = True
    
    # observation groups
    policy: PolicyCfg = PolicyCfg()

@configclass
class SpotEventCfg:
    """Configuration for randomization."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),#"body"
            "mass_distribution_params": (-0.05, 0.05),#(-2.5, 2.5) scaled down for 560g MP
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),#"body"
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3), #(-1.5, 1.5) OG Spot
                "y": (-0.2, 0.2), #(-1.0, 1.0) OG Spot
                "z": (-0.1, 0.1), #(-0.5, 0.5) OG Spot
                "roll": (-0.2, 0.2), #(-0.7, 0.7) OG Spot
                "pitch": (-0.2, 0.2), #(-0.7, 0.7) OG Spot
                "yaw": (-0.3, 0.3), #(-1.0, 1.0) OG Spot
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=spot_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-1.0, 1.0), # (-2.5, 2.5) OG Spot
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),#(10.0, 15.0)
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

    # randomize_actuator_gains = EventTerm(
    # func=mdp.randomize_actuator_gains,
    # mode="reset",
    # params={
    #     "asset_cfg": SceneEntityCfg("robot"),
    #     "stiffness_distribution_params": (0.7, 1.3),
    #     "damping_distribution_params": (0.6, 1.5),
    #     "operation": "scale",
    # },
    # )


@configclass
class SpotRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.25,
            "velocity_threshold": 0.15,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["lb3", "lf3", "rb3", "rf3"]),
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward,
        weight=10.0,
        params={"std": 0.2, "asset_cfg": SceneEntityCfg("robot")}, #taken donw from 0.2
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward,
        weight=15.0,
        params={
            "std": 0.1,
            "ramp_rate": 0.5,
            "ramp_at_vel": 0.15,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward,
        weight=1.0,#was 1.0 
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.03, #was 0.03
            "asset_cfg": SceneEntityCfg("robot", body_names=["lb3", "lf3", "rb3", "rf3"]),
        },
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward,
        weight=10.0,
        params={
            "std": 0.08,
            "max_err": 0.15,
            "velocity_threshold": 0.1,
            "synced_feet_pair_names": (("lf3", "rb3"), ("rf3", "lb3")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(
        func=spot_mdp.action_smoothness_penalty,
        weight=-2.0,#curriculum started with -1.0
    )
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["lb3", "lf3", "rb3", "rf3"])},
    )
    base_motion = RewardTermCfg(
        func=spot_mdp.base_motion_penalty,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    base_orientation = RewardTermCfg(
        func=spot_mdp.base_orientation_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["lb3", "lf3", "rb3", "rf3"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["lb3", "lf3", "rb3", "rf3"]),
            "threshold": 1.0,
        },
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty,
        weight=-2.5e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["base_lf1", "base_lb1", "base_rf1", "base_rb1"])},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "base_lf1", "lf1_lf2", "lf2_lf3",
                "base_rf1", "rf1_rf2", "rf2_rf3",
                "base_lb1", "lb1_lb2", "lb2_lb3",
                "base_rb1", "rb1_rb2", "rb2_rb3",
            ]),
            "stand_still_scale": 10.0, #curriculum started at 5.0
            "velocity_threshold": 0.15,
        },
    )

    # SpotRewardsCfg — add new term
    hip_abduction = RewardTermCfg(
        func=spot_mdp.joint_position_penalty,
        weight=-2.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
                "base_lf1", "base_rf1", "base_lb1", "base_rb1"
            ]),
            "stand_still_scale": 1.0,
            "velocity_threshold": 0.0,
        },
    )

    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "base_lf1", "lf1_lf2", "lf2_lf3",
            "base_rf1", "rf1_rf2", "rf2_rf3",
            "base_lb1", "lb1_lb2", "lb2_lb3",
            "base_rb1", "rb1_rb2", "rb2_rb3",
        ])},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["base_lf1", "base_rf1", "base_lb1", "base_rb1"])},
    )

@configclass
class SpotTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    # chassis
                    "base_link",
                    # leg segments
                    "lf1","lf2",
                    "lb1","lb2",
                    "rf1","rf2",
                    "rb1","rb2",
                ],
            ),
            "threshold": 1.0,
        },
    )  
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

@configclass
class SpotFlatEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings
    observations: SpotObservationsCfg = SpotObservationsCfg()
    actions: SpotActionsCfg = SpotActionsCfg()
    commands: SpotCommandsCfg = SpotCommandsCfg()

    # MDP setting
    rewards: SpotRewardsCfg = SpotRewardsCfg()
    terminations: SpotTerminationsCfg = SpotTerminationsCfg()
    events: SpotEventCfg = SpotEventCfg()

    # Viewer
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        
        # post init of parent
        super().__post_init__()
        # general settings
        self.decimation = 10  # 50 Hz
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.002  # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Spot-d
        self.scene.robot = CUSTOM_QUAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=True,
        )

        # no height scan
        self.scene.height_scanner = None

        # Check if there are any height-related observations
        if hasattr(self.observations.policy, 'height_scan'):
            delattr(self.observations.policy, 'height_scan')


class SpotFlatEnvCfg_PLAY(SpotFlatEnvCfg):
    commands: SpotCommandsCfg_PLAY = SpotCommandsCfg_PLAY()

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
            # remove random pushing event