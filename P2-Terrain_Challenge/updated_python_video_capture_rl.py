#!/usr/bin/env python3
"""
Fixed Mini Pupper RL Training with Proper GPU Usage
Addresses tensor creation, GPU utilization, and environment compatibility issues
"""

# ✅ Force GPU selection before any other imports
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Makes GPU 1 appear as cuda:0 to PyTorch

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
import gym
from gym import spaces
import signal


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


class MiniPupperEnv(gym.Env):
    """
    Fixed Gym-compatible environment for Mini Pupper with proper GPU support
    """
    def __init__(self, launch_file='main.launch.py'):
        super(MiniPupperEnv, self).__init__()
        
        # Launch Gazebo with proper environment
        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "1",
            "ROBOT_MODEL": "mini_pupper",
            "DISPLAY": "",
            "QT_QPA_PLATFORM": "offscreen",
        })
        
        self.launch_process = subprocess.Popen([
            "ros2", "launch", "mini_pupper_simulation", "main.launch.py",
            "world_init_z:=0.3",
            "world_init_heading:=3.14159",
            "gui:=false"
        ], env=env)
        
        print("🚀 Waiting for Gazebo...")
        time.sleep(8)  # Increased wait time
        
        # ROS 2 node setup
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node('minipupper_gym_env')
        
        # Publishers and subscribers
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_sub = self.node.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        
        # ✅ Fixed action and observation spaces to match training code
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        self.latest_joint_state = None
        self.step_count = 0
        self.max_steps = 200
    
    def joint_callback(self, msg):
        self.latest_joint_state = msg
    
    def reset(self):
        """Reset environment and return initial observation"""
        self.step_count = 0
        self.latest_joint_state = None
        print("🔄 Resetting environment")
        
        # Stop the robot
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        time.sleep(0.5)
        
        # Wait for initial joint state
        timeout = 10.0
        start_time = time.time()
        while rclpy.ok() and self.latest_joint_state is None:
            if time.time() - start_time > timeout:
                print("⚠️ Timeout waiting for joint states, using default")
                break
            rclpy.spin_once(self.node, timeout_sec=0.1)
        
        return self._get_obs()
    
    def step(self, action):
        """Execute action and return observation, reward, done"""
        self.step_count += 1
        
        # ✅ Handle 4D action space properly
        cmd = Twist()
        cmd.linear.x = float(action[0]) * 0.3
        cmd.angular.z = float(action[1]) * 1.0
        if len(action) > 2:
            cmd.linear.y = float(action[2]) * 0.2
        self.cmd_pub.publish(cmd)
        
        # Allow motion to propagate
        time.sleep(0.05)
        rclpy.spin_once(self.node, timeout_sec=0.01)
        
        obs = self._get_obs()
        
        # ✅ Better reward function for training
        balance_reward = -np.sum(np.abs(obs[:6])) * 2
        motion_reward = np.sum(np.abs(action)) * 0.1
        reward = balance_reward + motion_reward
        
        done = self.step_count >= self.max_steps or abs(balance_reward) > 15
        
        return obs, reward, done
    
    def _get_obs(self):
        """Get current observation"""
        if self.latest_joint_state is None:
            return np.zeros(12, dtype=np.float32)
        
        pos = np.array(self.latest_joint_state.position, dtype=np.float32)
        vel = np.array(self.latest_joint_state.velocity, dtype=np.float32)
        
        # Ensure we have exactly 12 dimensions
        if len(pos) < 6:
            pos = np.pad(pos, (0, 6 - len(pos)))
        if len(vel) < 6:
            vel = np.pad(vel, (0, 6 - len(vel)))
        
        return np.concatenate([pos[:6], vel[:6]])
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        """Clean shutdown"""
        if hasattr(self, 'node'):
            self.node.destroy_node()
        if hasattr(self, 'launch_process'):
            self.launch_process.send_signal(signal.SIGINT)
            self.launch_process.wait()
            print("🛑 Gazebo shutdown")


class PolicyNetwork(nn.Module):
    """Enhanced policy network with proper GPU utilization"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for better generalization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        # ✅ Initialize weights properly for GPU training
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state):
        return self.network(state)


class EnhancedRLTrainer:
    """RL trainer with fixed GPU utilization"""
    
    def __init__(self, env, device, video_recorder):
        self.env = env
        self.device = device
        self.video_recorder = video_recorder
        
        # ✅ Fixed policy network initialization
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3, weight_decay=1e-5)
        
        # Training metrics
        self.episode_rewards = []
        self.losses = []
        
        # ✅ Verify GPU usage
        if torch.cuda.is_available():
            # Create a test tensor to verify GPU is working
            test_tensor = torch.randn(100, 100).to(device)
            test_result = torch.mm(test_tensor, test_tensor.t())
            print(f"✅ GPU test successful - tensor on {test_result.device}")
        
        print(f"🤖 Enhanced RL Trainer initialized on {device}")
    
    def get_gpu_utilization(self):
        """Get GPU utilization for overlay"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                capture_output=True, text=True
            )
            gpu_utils = result.stdout.strip().split('\n')
            return int(gpu_utils[1]) if len(gpu_utils) > 1 else 0
        except:
            return 0
    
    def collect_trajectory(self):
        """Collect trajectory with proper tensor handling"""
        states, actions, rewards = [], [], []
        
        state = self.env.reset()
        done = False
        
        while not done:
            # ✅ Ensure state is float32 numpy array
            state = np.array(state, dtype=np.float32)
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.policy(state_tensor).cpu().numpy()[0]
            
            # Add exploration noise
            if np.random.random() < 0.1:
                action = np.random.uniform(-1, 1, self.env.action_space.shape[0])
            
            next_state, reward, done = self.env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            state = next_state
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
        """Training step with proper tensor creation"""
        # ✅ Fixed tensor creation - convert to numpy first, then to tensor
        states_np = np.array(states, dtype=np.float32)
        actions_np = np.array(actions, dtype=np.float32)
        returns_np = np.array(returns, dtype=np.float32)
        
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).to(self.device)
        returns = torch.from_numpy(returns_np).to(self.device)
        
        # Normalize returns
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / returns.std()
        
        # ✅ Policy gradient loss with proper GPU computation
        predicted_actions = self.policy(states)
        action_diff = predicted_actions - actions
        policy_loss = torch.mean(action_diff.pow(2), dim=1)
        loss = torch.mean(returns * policy_loss)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # ✅ Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def train_with_video(self, num_episodes=30):
        """Train with video recording and proper GPU usage"""
        print(f"🎬 Starting video training for {num_episodes} episodes...")
        
        # Start recording
        if not self.video_recorder.start_recording():
            print("❌ Failed to start recording, continuing without video...")
        
        try:
            for episode in range(num_episodes):
                # Collect trajectory
                states, actions, rewards = self.collect_trajectory()
                
                if len(states) == 0:
                    print(f"⚠️ Episode {episode}: No states collected, skipping...")
                    continue
                
                # Compute returns and train
                returns = self.compute_returns(rewards)
                loss = self.train_step(states, actions, returns)
                
                # Track metrics
                total_reward = sum(rewards)
                self.episode_rewards.append(total_reward)
                self.losses.append(loss)
                
                # Get GPU utilization
                gpu_util = self.get_gpu_utilization()
                
                # Print progress
                if episode % 5 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else total_reward
                    print(f"🎯 Episode {episode:3d} | Reward: {total_reward:6.2f} | Avg: {avg_reward:6.2f} | Loss: {loss:6.4f} | GPU: {gpu_util}%")
                
                time.sleep(0.5)
            
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
        except Exception as e:
            print(f"❌ Training error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.video_recorder.stop_recording()
            print("✅ Training and recording completed!")
    
    def demonstrate_policy(self, steps=50):
        """Demonstrate the trained policy"""
        print("\n🎭 Demonstrating trained policy...")
        
        state = self.env.reset()
        total_reward = 0
        
        for step in range(steps):
            state = np.array(state, dtype=np.float32)
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                action = self.policy(state_tensor).cpu().numpy()[0]
            
            state, reward, done = self.env.step(action)
            total_reward += reward
            
            if done:
                state = self.env.reset()
            
            time.sleep(0.05)
        
        print(f"🏆 Demo reward: {total_reward:.2f}")


def main():
    """Main function with fixed GPU training"""
    print("🎬 Mini Pupper RL Training with Fixed GPU Usage")
    print("=" * 60)
    
    # ✅ Verify CUDA setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Training device: {device}")
    
    if torch.cuda.is_available():
        print(f"📈 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"📍 CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        
        # ✅ Test GPU computation
        test_tensor = torch.randn(1000, 1000).to(device)
        start_time = time.time()
        result = torch.mm(test_tensor, test_tensor)
        gpu_time = time.time() - start_time
        print(f"⚡ GPU matrix multiplication test: {gpu_time:.4f}s")
    
    # Initialize components
    video_recorder = VideoRecorder()
    
    try:
        env = MiniPupperEnv()
        trainer = EnhancedRLTrainer(env, device, video_recorder)
        
        print("\n🎬 Recording will start in 3 seconds...")
        print("🎯 Make sure your screen is ready for recording!")
        time.sleep(3)
        
        # Train with video recording
        trainer.train_with_video(num_episodes=30)
        
        # Demonstrate final policy
        print("\n🎭 Recording final demonstration...")
        trainer.demonstrate_policy(steps=50)
        
    except Exception as e:
        print(f"❌ Main execution error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        try:
            if 'env' in locals():
                env.close()
            if rclpy.ok():
                rclpy.shutdown()
        except:
            pass
    
    print(f"\n🎉 Complete! Check your video: {video_recorder.video_path}")
    print("💡 Tip: You can speed up the video with:")
    print(f"   ffmpeg -i {video_recorder.video_path} -filter:v 'setpts=0.5*PTS' output_2x_speed.mp4")


if __name__ == "__main__":
    main()