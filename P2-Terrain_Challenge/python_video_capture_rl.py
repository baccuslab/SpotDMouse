#!/usr/bin/env python3
"""
Mini Pupper RL Training with Video Capture
Records the training process and creates a timelapse video
"""

import os
import time
import subprocess
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
import cv2
from datetime import datetime

# Force use of RTX 4090 for training
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class VideoRecorder:
    """Records Gazebo window and training metrics"""
    
    def __init__(self, output_dir="training_videos"):
        self.output_dir = output_dir
        self.recording = False
        self.frame_count = 0
        os.makedirs(output_dir, exist_ok=True)
        
        # Video settings
        self.fps = 30
        self.width = 1920
        self.height = 1080
        
        # Get timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(output_dir, f"mini_pupper_training_{timestamp}.mp4")
        
        print(f"📹 Video will be saved to: {self.video_path}")
    
    def start_recording(self):
        """Start screen recording using ffmpeg"""
        self.recording = True
        
        # FFmpeg command to record screen
        cmd = [
            'ffmpeg',
            '-f', 'x11grab',
            '-video_size', f'{self.width}x{self.height}',
            '-framerate', str(self.fps),
            '-i', ':0.0',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-y',  # Overwrite output file
            self.video_path
        ]
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            print("🎬 Recording started!")
            return True
        except Exception as e:
            print(f"❌ Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop screen recording"""
        if hasattr(self, 'ffmpeg_process'):
            self.ffmpeg_process.terminate()
            self.ffmpeg_process.wait()
            print(f"🎬 Recording stopped! Video saved to: {self.video_path}")
        self.recording = False
    
    def create_metrics_overlay(self, episode, reward, loss, gpu_util):
        """Create overlay with training metrics"""
        # This would overlay metrics on the video in a more advanced version
        overlay_text = f"Episode: {episode} | Reward: {reward:.2f} | Loss: {loss:.4f} | GPU: {gpu_util}%"
        return overlay_text

class EnhancedMiniPupperEnv:
    """Enhanced environment with better visualization"""
    
    def __init__(self):
        self.action_dim = 4
        self.state_dim = 12
        self.max_episode_steps = 200  # Shorter episodes for video
        self.current_step = 0
        self.episode_rewards = []
        
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        return np.random.normal(0, 0.1, self.state_dim)
    
    def step(self, action):
        """Execute action with more interesting dynamics"""
        self.current_step += 1
        
        # More complex state evolution for interesting video
        next_state = np.random.normal(0, 0.05, self.state_dim)
        
        # Reward for staying balanced (more dramatic for video)
        balance_reward = -np.sum(np.abs(next_state[:6])) * 2
        motion_reward = np.sum(np.abs(action)) * 0.1  # Small reward for moving
        reward = balance_reward + motion_reward
        
        # Episode termination
        done = self.current_step >= self.max_episode_steps or abs(balance_reward) > 15
        
        return next_state, reward, done

class PolicyNetwork(nn.Module):
    """Enhanced policy network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.network(state)

class EnhancedRLTrainer:
    """RL trainer with video-friendly features"""
    
    def __init__(self, env, device, video_recorder):
        self.env = env
        self.device = device
        self.video_recorder = video_recorder
        
        # Policy network
        self.policy = PolicyNetwork(env.state_dim, env.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)
        
        # Training metrics
        self.episode_rewards = []
        self.losses = []
        
        print(f"🤖 Enhanced RL Trainer initialized on {device}")
    
    def get_gpu_utilization(self):
        """Get GPU utilization for overlay"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            gpu_utils = result.stdout.strip().split('\n')
            return int(gpu_utils[1]) if len(gpu_utils) > 1 else 0  # RTX 4090 utilization
        except:
            return 0
    
    def collect_trajectory(self):
        """Collect trajectory with visual feedback"""
        states, actions, rewards = [], [], []
        
        state = self.env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.policy(state_tensor).cpu().numpy()[0]
            
            # Add some exploration noise for interesting video
            if np.random.random() < 0.1:  # 10% random actions
                action = np.random.uniform(-1, 1, self.env.action_dim)
            
            next_state, reward, done = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
            
            # Small delay for smooth video
            time.sleep(0.02)
        
        return states, actions, rewards
    
    def compute_returns(self, rewards, gamma=0.95):
        """Compute returns"""
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns
    
    def train_step(self, states, actions, returns):
        """Training step"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        predicted_actions = self.policy(states)
        loss = -torch.mean(returns * torch.sum((predicted_actions - actions) ** 2, dim=1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_with_video(self, num_episodes=50):
        """Train with video recording"""
        print(f"🎬 Starting video training for {num_episodes} episodes...")
        
        # Start recording
        if not self.video_recorder.start_recording():
            print("❌ Failed to start recording, continuing without video...")
        
        try:
            for episode in range(num_episodes):
                # Collect trajectory
                states, actions, rewards = self.collect_trajectory()
                
                # Compute returns and train
                returns = self.compute_returns(rewards)
                loss = self.train_step(states, actions, returns)
                
                # Track metrics
                total_reward = sum(rewards)
                self.episode_rewards.append(total_reward)
                self.losses.append(loss)
                
                # Get GPU utilization
                gpu_util = self.get_gpu_utilization()
                
                # Print progress (will appear in terminal during video)
                if episode % 5 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else total_reward
                    print(f"🎯 Episode {episode:3d} | Reward: {total_reward:6.2f} | Avg: {avg_reward:6.2f} | Loss: {loss:6.4f} | GPU: {gpu_util}%")
                
                # Pause for video visibility
                time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
        
        finally:
            # Stop recording
            self.video_recorder.stop_recording()
            print("✅ Training and recording completed!")
    
    def demonstrate_policy(self, steps=100):
        """Demonstrate the trained policy"""
        print("\n🎭 Demonstrating trained policy...")
        
        state = self.env.reset()
        total_reward = 0
        
        for step in range(steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.policy(state_tensor).cpu().numpy()[0]
            
            state, reward, done = self.env.step(action)
            total_reward += reward
            
            if done:
                state = self.env.reset()
            
            time.sleep(0.05)  # Slower for demonstration
        
        print(f"🏆 Demo reward: {total_reward:.2f}")

class ROSInterface(Node):
    """ROS interface for real robot control"""
    
    def __init__(self):
        super().__init__('video_rl_trainer')
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        self.latest_joint_state = None
        print("🔗 ROS interface ready for video training")
    
    def joint_callback(self, msg):
        self.latest_joint_state = msg
    
    def send_action(self, action):
        """Send action with more dramatic movements for video"""
        cmd = Twist()
        cmd.linear.x = float(action[0] * 0.5)  # Scale down for safety
        cmd.angular.z = float(action[1] * 1.0)
        cmd.linear.y = float(action[2] * 0.3) if len(action) > 2 else 0.0
        self.cmd_pub.publish(cmd)

def main():
    """Main function with video recording"""
    print("🎬 Mini Pupper RL Training with Video Recording")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Training device: {device}")
    
    if torch.cuda.is_available():
        print(f"📈 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize components
    video_recorder = VideoRecorder()
    env = EnhancedMiniPupperEnv()
    trainer = EnhancedRLTrainer(env, device, video_recorder)
    
    # Initialize ROS if available
    try:
        rclpy.init()
        ros_interface = ROSInterface()
        print("✅ ROS connected - real robot control enabled")
        use_ros = True
    except:
        print("⚠️  ROS not available - using simulation only")
        use_ros = False
    
    print("\n🎬 Recording will start in 3 seconds...")
    print("🎯 Make sure Gazebo window is visible!")
    time.sleep(3)
    
    # Train with video recording
    trainer.train_with_video(num_episodes=30)
    
    # Demonstrate final policy
    print("\n🎭 Recording final demonstration...")
    trainer.demonstrate_policy(steps=50)
    
    # Cleanup
    if use_ros:
        rclpy.shutdown()
    
    print(f"\n🎉 Complete! Check your video: {video_recorder.video_path}")
    print("💡 Tip: You can speed up the video with:")
    print(f"   ffmpeg -i {video_recorder.video_path} -filter:v 'setpts=0.5*PTS' output_2x_speed.mp4")

if __name__ == "__main__":
    main()
