# isaac_minipupper_training.py
"""
Mini Pupper Isaac Lab Training Environment
Converts ROS2/Gazebo training to Isaac Lab for terrain traversal
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces

# Isaac Lab imports
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.envs import BaseEnv, BaseEnvCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import BaseScene, SceneEntityCfg
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR

# Import existing quadruped config as base
from omni.isaac.lab_assets import ANYMAL_C_CFG


@configclass
class MiniPupperSceneCfg(BaseScene):
    """Configuration for Mini Pupper scene with terrain obstacles"""
    
    # Ground plane with procedural terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=dict(
            curriculum=True,
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            # Add different terrain types for obstacles
            sub_terrains={
                "pyramid_stairs": (0.2, 0.0),
                "pyramid_stairs_inv": (0.2, 0.0), 
                "discrete_obstacles": (0.2, 0.0),
                "wave": (0.1, 0.0),
                "stairs": (0.1, 0.0),
                "pyramid": (0.1, 0.0),
                "random_uniform": (0.1, 0.0),
            }
        ),
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAAC_NUCLEUS_DIR}/Materials/TilesArabescato/TilesArabescato.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    
    # Mini Pupper robot (using Anymal as base, can be customized)
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            # Use Anymal for now - you can replace with Mini Pupper USD later
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_c/anymal_c.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.42),
            joint_pos={
                ".*HAA": 0.0,  # Hip abduction
                ".*HFE": 0.4,  # Hip flexion
                ".*KFE": -0.8, # Knee flexion
            },
            joint_vel={".*": 0.0},
        ),
        actuators={
            "base_legs": sim_utils.DCMotorCfg(
                joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
                effort_limit=40.0,
                saturation_effort=40.0,
                velocity_limit=10.0,
                stiffness=25.0,
                damping=0.5,
            ),
        },
    )
    
    # Contact sensors for feet
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT", 
        history_length=3, 
        track_air_time=True
    )
    
    # Height scanner for terrain awareness
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=RayCasterCfg.PatternCfg("grid", resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass 
class MiniPupperEnvCfg(BaseEnvCfg):
    """Configuration for Mini Pupper training environment"""
    
    # Scene settings
    scene: MiniPupperSceneCfg = MiniPupperSceneCfg(num_envs=4096, env_spacing=2.5)
    
    # Basic settings
    decimation = 4  # Control frequency = sim_frequency / decimation
    episode_length_s = 20.0
    
    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1/120,  # 120 Hz simulation
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply", 
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    
    # Viewer settings (for debugging)
    viewer: ViewerCfg = ViewerCfg(
        eye=[7.5, 7.5, 7.5], 
        lookat=[0.0, 0.0, 0.0]
    )


class MiniPupperEnv(BaseEnv):
    """Mini Pupper training environment for terrain traversal"""
    
    cfg: MiniPupperEnvCfg
    
    def __init__(self, cfg: MiniPupperEnvCfg, render_mode: str | None = None, **kwargs):
        # Initialize observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32  # 12 joint targets
        )
        
        # Initialize base environment
        super().__init__(cfg, render_mode, **kwargs)
        
        # Joint targets and previous actions
        self.joint_targets = torch.zeros(self.num_envs, 12, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, 12, device=self.device)
        
        # Rewards tracking
        self.episode_rewards = torch.zeros(self.num_envs, device=self.device)
        
        print(f"🤖 Mini Pupper environment initialized with {self.num_envs} environments")
    
    def _setup_scene(self):
        """Setup the scene entities"""
        # Add robot
        self.cfg.scene.robot.prim_path = self.template_env_ns + "/Robot"
        self.scene.articulations["robot"] = self.cfg.scene.robot
        
        # Add sensors
        self.cfg.scene.contact_forces.prim_path = self.template_env_ns + "/Robot/.*_FOOT"
        self.scene.sensors["contact_forces"] = self.cfg.scene.contact_forces
        
        self.cfg.scene.height_scanner.prim_path = self.template_env_ns + "/Robot/base"
        self.scene.sensors["height_scanner"] = self.cfg.scene.height_scanner
        
        # Clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.scene.terrain.prim_path])
        
        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
    
    def _pre_physics_step(self, actions: torch.Tensor):
        """Apply actions to the robot"""
        # Scale actions to joint limits
        self.joint_targets = actions * 1.0  # Scale as needed
        self.scene.articulations["robot"].set_joint_position_target(self.joint_targets)
        self.prev_actions = actions.clone()
    
    def _apply_action(self):
        """Apply the actions to the environment entities."""
        pass  # Already handled in _pre_physics_step
    
    def _get_observations(self) -> dict:
        """Compute observations"""
        # Get robot state
        robot = self.scene.articulations["robot"]
        
        # Base orientation (roll, pitch, yaw)
        base_quat = robot.data.root_quat_w
        base_rpy = quat_to_euler_xyz(base_quat)
        
        # Base angular velocity
        base_ang_vel = robot.data.root_ang_vel_b
        
        # Joint positions and velocities (12 joints)
        joint_pos = robot.data.joint_pos
        joint_vel = robot.data.joint_vel
        
        # Contact forces (4 feet)
        contact_forces = self.scene.sensors["contact_forces"].data.net_forces_w
        contact_binary = torch.norm(contact_forces, dim=-1) > 1.0
        
        # Height scan (terrain awareness)
        height_data = self.scene.sensors["height_scanner"].data.ray_hits_w[..., 2]
        height_mean = torch.mean(height_data, dim=1, keepdim=True)
        
        # Previous actions
        prev_actions = self.prev_actions
        
        # Combine all observations (3 + 3 + 12 + 12 + 4 + 1 + 12 = 47, pad to 48)
        obs = torch.cat([
            base_rpy,              # 3
            base_ang_vel,          # 3  
            joint_pos,             # 12
            joint_vel,             # 12
            contact_binary.float(), # 4
            height_mean,           # 1
            prev_actions,          # 12
            torch.zeros(self.num_envs, 1, device=self.device)  # 1 (padding)
        ], dim=-1)
        
        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for terrain traversal"""
        robot = self.scene.articulations["robot"]
        
        # Forward velocity reward
        base_vel = robot.data.root_lin_vel_b
        forward_vel = base_vel[:, 0]  # X-axis forward
        forward_reward = torch.clamp(forward_vel, 0.0, 2.0)
        
        # Stability reward (penalize excessive roll/pitch)
        base_quat = robot.data.root_quat_w
        base_rpy = quat_to_euler_xyz(base_quat)
        stability_reward = -torch.abs(base_rpy[:, 0]) - torch.abs(base_rpy[:, 1])  # roll + pitch
        
        # Contact reward (encourage proper gait)
        contact_forces = self.scene.sensors["contact_forces"].data.net_forces_w
        contact_binary = torch.norm(contact_forces, dim=-1) > 1.0
        contact_reward = torch.sum(contact_binary.float(), dim=1) * 0.1
        
        # Action smoothness (penalize large action changes)
        if hasattr(self, 'prev_actions'):
            action_diff = torch.sum((self.joint_targets - self.prev_actions) ** 2, dim=1)
            smoothness_reward = -action_diff * 0.01
        else:
            smoothness_reward = torch.zeros(self.num_envs, device=self.device)
        
        # Height maintenance (stay at reasonable height)
        base_height = robot.data.root_pos_w[:, 2]
        height_reward = -torch.abs(base_height - 0.42) * 2.0
        
        # Total reward
        total_reward = (forward_reward * 1.0 + 
                       stability_reward * 0.5 + 
                       contact_reward * 0.2 + 
                       smoothness_reward * 0.1 +
                       height_reward * 0.3)
        
        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if episodes are done"""
        robot = self.scene.articulations["robot"]
        
        # Terminal conditions
        base_height = robot.data.root_pos_w[:, 2]
        base_quat = robot.data.root_quat_w
        base_rpy = quat_to_euler_xyz(base_quat)
        
        # Terminate if robot falls or tips over
        height_term = base_height < 0.2
        orientation_term = (torch.abs(base_rpy[:, 0]) > 1.57) | (torch.abs(base_rpy[:, 1]) > 1.57)
        
        terminated = height_term | orientation_term
        truncated = self.episode_length_buf >= self.max_episode_length
        
        return terminated, truncated
    
    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset specified environments"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # Reset robot state
        robot = self.scene.articulations["robot"]
        
        # Random initial positions on terrain
        pos_x = torch.uniform(-1.0, 1.0, (len(env_ids),), device=self.device) 
        pos_y = torch.uniform(-1.0, 1.0, (len(env_ids),), device=self.device)
        pos_z = torch.full((len(env_ids),), 0.42, device=self.device)
        
        # Set initial state
        robot.set_world_poses(
            torch.stack([pos_x, pos_y, pos_z], dim=1),
            robot.data.default_root_quat[env_ids],
            env_ids
        )
        robot.set_joint_positions(robot.data.default_joint_pos[env_ids], env_ids)
        robot.set_joint_velocities(robot.data.default_joint_vel[env_ids], env_ids)
        
        # Reset tracking variables
        self.joint_targets[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.episode_rewards[env_ids] = 0.0
        
        super()._reset_idx(env_ids)


class PolicyNetwork(nn.Module):
    """Policy network for Mini Pupper control"""
    
    def __init__(self, obs_dim=48, action_dim=12, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, obs):
        return self.network(obs)


class MiniPupperTrainer:
    """RL trainer for Mini Pupper terrain traversal"""
    
    def __init__(self, env, device):
        self.env = env
        self.device = device
        
        # Initialize policy
        self.policy = PolicyNetwork().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        # Training metrics
        self.episode_rewards = []
        self.losses = []
        
        print(f"🎯 Mini Pupper trainer initialized on {device}")
    
    def collect_rollouts(self, num_steps=1000):
        """Collect training data"""
        observations = []
        actions = []
        rewards = []
        
        obs = self.env.reset()[0]["policy"]
        
        for step in range(num_steps):
            # Get action from policy
            with torch.no_grad():
                action = self.policy(obs)
                
            # Add exploration noise
            if np.random.random() < 0.1:
                noise = torch.randn_like(action) * 0.1
                action = torch.clamp(action + noise, -1.0, 1.0)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            observations.append(obs.cpu())
            actions.append(action.cpu())
            rewards.append(reward.cpu())
            
            obs = next_obs["policy"]
            
            # Reset if needed
            if terminated.any() or truncated.any():
                obs = self.env.reset()[0]["policy"]
        
        return observations, actions, rewards
    
    def train_step(self, observations, actions, rewards):
        """Perform training step"""
        # Convert to tensors
        obs_tensor = torch.stack(observations).to(self.device)
        action_tensor = torch.stack(actions).to(self.device)
        reward_tensor = torch.stack(rewards).to(self.device)
        
        # Compute returns (simple approach)
        returns = []
        R = 0
        for r in reversed(reward_tensor):
            R = r + 0.99 * R
            returns.insert(0, R)
        returns = torch.stack(returns)
        
        # Normalize returns
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()
        
        # Policy loss
        pred_actions = self.policy(obs_tensor.view(-1, obs_tensor.shape[-1]))
        action_diff = pred_actions - action_tensor.view(-1, action_tensor.shape[-1])
        policy_loss = torch.mean(action_diff.pow(2), dim=1)
        loss = torch.mean(returns.view(-1) * policy_loss)
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_iterations=100):
        """Main training loop"""
        print(f"🚀 Starting Mini Pupper terrain training...")
        
        for iteration in range(num_iterations):
            # Collect data
            observations, actions, rewards = self.collect_rollouts(1000)
            
            # Train
            loss = self.train_step(observations, actions, rewards)
            
            # Track metrics
            avg_reward = torch.stack(rewards).mean().item()
            self.episode_rewards.append(avg_reward)
            self.losses.append(loss)
            
            if iteration % 10 == 0:
                print(f"Iteration {iteration:3d} | Avg Reward: {avg_reward:6.2f} | Loss: {loss:6.4f}")
        
        print("✅ Training completed!")


def main():
    """Main training script"""
    print("🐕 Mini Pupper Isaac Lab Training")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create environment
    env_cfg = MiniPupperEnvCfg()
    env = MiniPupperEnv(env_cfg)
    
    # Create trainer
    trainer = MiniPupperTrainer(env, device)
    
    # Train
    trainer.train(num_iterations=200)
    
    # Save model
    torch.save(trainer.policy.state_dict(), "mini_pupper_policy.pth")
    print("💾 Model saved!")
    
    env.close()


if __name__ == "__main__":
    main()