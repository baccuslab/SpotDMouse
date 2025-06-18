# gym_ros_minipupper.py
import os
import gym
import rclpy
import subprocess
import signal
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from gym import spaces
import time
import os
import subprocess


class MiniPupperEnv(gym.Env):
    """
    Gym-compatible environment that wraps ROS 2 and Gazebo for the Mini Pupper robot.
    Assumes Gazebo has been launched with appropriate topics (`/cmd_vel`, `/joint_states`).
    """
    def __init__(self, launch_file='bringup.launch.py'):
        super(MiniPupperEnv, self).__init__()

        # Launch Gazebo headless
        # self.launch_process = subprocess.Popen([
        #     'ros2', 'launch', 'mini_pupper_gazebo', launch_file, 'gui:=false'
        # ])

        env = os.environ.copy()
        env.update({
            "CUDA_VISIBLE_DEVICES": "1",                   # ✅ use GPU 1 (e.g., 4090)
            "ROBOT_MODEL": "mini_pupper",                  # ✅ required by your sim
            "DISPLAY": "",                                  # ✅ run Gazebo fully headless
            "QT_QPA_PLATFORM": "offscreen",                # ✅ prevent Qt-based GUIs
        })

        self.launch_process = subprocess.Popen([
            "ros2", "launch", "mini_pupper_simulation", "main.launch.py",
            "world_init_z:=0.3",
            "world_init_heading:=3.14159",
            "gui:=false"                                   # ✅ if supported by launch file
        ], env=env)
        # Wait for Gazebo to initialize
        print("🚀 Waiting for Gazebo...")
        time.sleep(5)

        # ROS 2 node setup
        rclpy.init()
        self.node = rclpy.create_node('minipupper_gym_env')

        # Publishers and subscribers
        self.cmd_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.joint_sub = self.node.create_subscription(JointState, '/joint_states', self.joint_callback, 10)

        # Action: linear.x, angular.z (normalized between -1 and 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observation: 12D joint state (6 pos, 6 vel) placeholder
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.latest_joint_state = None
        self.step_count = 0
        self.max_steps = 200

    def joint_callback(self, msg):
        self.latest_joint_state = msg

    def reset(self):
        self.step_count = 0

        # Call reset service if needed
        self.latest_joint_state = None
        print("🔄 Resetting environment")
        time.sleep(1.0)

        # Wait for initial joint state
        while rclpy.ok() and self.latest_joint_state is None:
            rclpy.spin_once(self.node)

        return self._get_obs()

    def step(self, action):
        self.step_count += 1

        cmd = Twist()
        cmd.linear.x = float(action[0]) * 0.3
        cmd.angular.z = float(action[1]) * 1.0
        self.cmd_pub.publish(cmd)

        time.sleep(0.1)  # Allow motion to propagate
        rclpy.spin_once(self.node)

        obs = self._get_obs()
        reward = -np.sum(np.abs(obs[:6]))  # Penalize joint displacement
        done = self.step_count >= self.max_steps
        print(f"🔧 Sending action: {action}")

        return obs, reward, done#, {}

    def _get_obs(self):
        if self.latest_joint_state is None:
            return np.zeros(12)
        pos = np.array(self.latest_joint_state.position)
        vel = np.array(self.latest_joint_state.velocity)
        return np.concatenate([pos[:6], vel[:6]])

    def render(self, mode='human'):
        pass  # No rendering in headless mode

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()
        if self.launch_process:
            self.launch_process.send_signal(signal.SIGINT)
            self.launch_process.wait()
            print("🛑 Gazebo shutdown")
