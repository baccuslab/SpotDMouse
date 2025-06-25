#!/usr/bin/env python3
"""
MiniPupper RL Training Script
"""

import numpy as np
import mujoco
import mujoco.viewer

class MiniPupperEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path("mini_pupper_mujoco.xml")
        self.data = mujoco.MjData(self.model)
        self.initial_qpos = self.data.qpos.copy()
        
    def reset(self):
        self.data.qpos[:] = self.initial_qpos
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
    
    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[7:],  # Joint positions (skip base position/orientation)
            self.data.qvel[6:],  # Joint velocities (skip base linear/angular velocity)
        ])
    
    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        reward = self.data.qvel[0]  # Forward velocity reward
        done = self.data.qpos[2] < 0.1  # Robot fell
        
        return obs, reward, done, {}
    
    def render(self):
        # For visualization
        pass

def test_random_policy():
    """Test with random actions"""
    env = MiniPupperEnv()
    
    print("Testing random policy...")
    
    obs = env.reset()
    total_reward = 0
    
    for step in range(1000):
        # Random actions
        action = np.random.uniform(-1, 1, size=env.model.nu)
        
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        
        if step % 100 == 0:
            print(f"Step {step}: reward={reward:.3f}, height={env.data.qpos[2]:.3f}")
        
        if done:
            print(f"Episode ended at step {step}")
            obs = env.reset()
            total_reward = 0
    
    print(f"Random policy test completed!")

def view_model():
    """View the model interactively"""
    try:
        model = mujoco.MjModel.from_xml_path("mini_pupper_mujoco.xml")
        data = mujoco.MjData(model)
        
        print("Opening MuJoCo viewer...")
        print("Controls:")
        print("  - Left click + drag: rotate view")
        print("  - Right click + drag: zoom")
        print("  - Scroll: zoom")
        print("  - Space: pause/unpause")
        print("  - ESC: exit")
        
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # Run simulation
            while viewer.is_running():
                # Apply some random actions to see movement
                data.ctrl[:] = 0.1 * np.sin(0.01 * data.time * np.arange(model.nu))
                
                mujoco.mj_step(model, data)
                viewer.sync()
                
    except Exception as e:
        print(f"Viewer error: {e}")
        print("Try: pip install mujoco[viewer]")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "view":
        view_model()
    else:
        test_random_policy()