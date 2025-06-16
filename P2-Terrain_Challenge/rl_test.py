#!/usr/bin/env python3
"""
Mini Pupper RL Training Script
Demonstrates simultaneous Gazebo simulation (RTX 2080) + RL training (RTX 4090)
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

# Force use of specific GPU for training
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use first RTX 4090

class MiniPupperEnv:
    """Simple RL environment for Mini Pupper balance task"""
    
    def __init__(self):
        self.action_dim = 4  # 4 leg actions (simplified)
        self.state_dim = 12  # joint positions + velocities
        self.max_episode_steps = 1000
        self.current_step = 0
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        # Return random initial state (in real implementation, get from Gazebo)
        return np.random.normal(0, 0.1, self.state_dim)
    
    def step(self, action):
        """Execute action and return next state, reward, done"""
        self.current_step += 1
        
        # Simulate next state (in real implementation, send to Gazebo and get response)
        next_state = np.random.normal(0, 0.1, self.state_dim)
        
        # Simple reward: negative distance from upright position
        reward = -np.sum(np.abs(next_state[:6]))  # Penalty for joint deviation
        
        # Episode ends if robot falls or max steps reached
        done = self.current_step >= self.max_episode_steps or abs(reward) > 10
        
        return next_state, reward, done

class PolicyNetwork(nn.Module):
    """Simple policy network for RL agent"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions between -1 and 1
        )
        
    def forward(self, state):
        return self.network(state)

class RLTrainer:
    """Simple RL trainer using policy gradients"""
    
    def __init__(self, env, device):
        self.env = env
        self.device = device
        
        # Initialize policy network
        self.policy = PolicyNetwork(env.state_dim, env.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        
        print(f"🤖 RL Trainer initialized on device: {device}")
        print(f"📊 Policy network parameters: {sum(p.numel() for p in self.policy.parameters()):,}")
    
    def collect_trajectory(self):
        """Collect one episode trajectory"""
        states, actions, rewards = [], [], []
        
        state = self.env.reset()
        done = False
        
        while not done:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action = self.policy(state_tensor).cpu().numpy()[0]
            
            # Execute action in environment
            next_state, reward, done = self.env.step(action)
            
            # Store trajectory
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
        
        return states, actions, rewards
    
    def compute_returns(self, rewards, gamma=0.99):
        """Compute discounted returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns
    
    def train_step(self, states, actions, returns):
        """Perform one training step"""
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Forward pass
        predicted_actions = self.policy(states)
        
        # Policy gradient loss (simplified)
        loss = -torch.mean(returns * torch.sum((predicted_actions - actions) ** 2, dim=1))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, num_episodes=100):
        """Train the policy"""
        print(f"🚀 Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            # Collect trajectory
            states, actions, rewards = self.collect_trajectory()
            
            # Compute returns
            returns = self.compute_returns(rewards)
            
            # Train
            loss = self.train_step(states, actions, returns)
            
            # Log progress
            total_reward = sum(rewards)
            if episode % 10 == 0:
                print(f"Episode {episode:3d} | Reward: {total_reward:6.2f} | Loss: {loss:6.4f}")
        
        print("✅ Training completed!")

class ROSInterface(Node):
    """ROS interface for communicating with Gazebo"""
    
    def __init__(self):
        super().__init__('rl_trainer_node')
        
        # Publishers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_pub = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        
        self.latest_joint_state = None
        print("🔗 ROS interface initialized")
    
    def joint_callback(self, msg):
        """Callback for joint state updates"""
        self.latest_joint_state = msg
    
    def send_action(self, action):
        """Send action to robot"""
        # Convert RL action to robot commands
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_pub.publish(cmd)
    
    def get_state(self):
        """Get current robot state"""
        if self.latest_joint_state is None:
            return np.zeros(12)  # Default state
        
        # Extract positions and velocities
        positions = np.array(self.latest_joint_state.position[:6])
        velocities = np.array(self.latest_joint_state.velocity[:6])
        return np.concatenate([positions, velocities])

def main():
    """Main training loop"""
    print("🐕 Mini Pupper RL Training Starting...")
    print("=" * 50)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Training device: {device}")
    
    if torch.cuda.is_available():
        print(f"📈 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test GPU computation
        test_tensor = torch.randn(1000, 1000).to(device)
        start_time = time.time()
        result = torch.matmul(test_tensor, test_tensor)
        gpu_time = time.time() - start_time
        print(f"⚡ GPU Compute Test: {gpu_time:.4f}s")
    
    # Initialize ROS (comment out if running without Gazebo)
    try:
        rclpy.init()
        ros_interface = ROSInterface()
        print("✅ ROS connection established")
        use_ros = True
    except Exception as e:
        print(f"⚠️  ROS not available: {e}")
        print("🔧 Running in simulation mode...")
        use_ros = False
    
    # Initialize environment and trainer
    env = MiniPupperEnv()
    trainer = RLTrainer(env, device)
    
    # Train the policy
    trainer.train(num_episodes=50)
    
    # Demonstrate trained policy
    print("\n🎯 Demonstrating trained policy...")
    state = env.reset()
    total_reward = 0
    
    for step in range(100):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = trainer.policy(state_tensor).cpu().numpy()[0]
        
        # Send action to robot if ROS is available
        if use_ros:
            ros_interface.send_action(action)
        
        state, reward, done = env.step(action)
        total_reward += reward
        
        if done:
            break
        
        time.sleep(0.01)  # Small delay for real-time demo
    
    print(f"🏆 Demo completed! Total reward: {total_reward:.2f}")
    
    # Cleanup
    if use_ros:
        rclpy.shutdown()
    
    print("🎉 Training and demo completed successfully!")

if __name__ == "__main__":
    main()
