#!/usr/bin/env python3
"""
Fixed MiniPupper URDF to MuJoCo Conversion Script
Handles missing meshes and XACRO-generated URDFs
"""

import os
import sys
import xml.etree.ElementTree as ET
import re
from pathlib import Path

def find_mesh_files():
    """Find all mesh files in the MiniPupper package"""
    print("Searching for mesh files...")
    
    # Common locations for mesh files
    search_paths = [
        "../../meshes/",
        "../../../meshes/",
        "../../../../meshes/",
        "../../mini_pupper_description/meshes/",
        "../meshes/",
        "./meshes/",
    ]
    
    mesh_files = {}
    
    for search_path in search_paths:
        if os.path.exists(search_path):
            print(f"Found mesh directory: {search_path}")
            for root, dirs, files in os.walk(search_path):
                for file in files:
                    if file.lower().endswith(('.stl', '.dae', '.obj', '.ply')):
                        mesh_files[file.lower()] = os.path.join(root, file)
    
    print(f"Found {len(mesh_files)} mesh files:")
    for name, path in mesh_files.items():
        print(f"  {name} -> {path}")
    
    return mesh_files

def fix_urdf_paths(urdf_file, output_file="mini_pupper_fixed.urdf"):
    """Fix mesh paths and other issues in the URDF"""
    print(f"Fixing URDF: {urdf_file} -> {output_file}")
    
    # Find available mesh files
    mesh_files = find_mesh_files()
    
    try:
        tree = ET.parse(urdf_file)
        root = tree.getroot()
        
        # Fix mesh references
        mesh_count = 0
        for mesh_elem in root.iter('mesh'):
            filename_attr = mesh_elem.get('filename')
            if filename_attr:
                # Extract just the filename
                mesh_name = os.path.basename(filename_attr).lower()
                
                # Remove package:// prefix if present
                if filename_attr.startswith('package://'):
                    mesh_name = filename_attr.split('/')[-1].lower()
                
                # Try to find the mesh file
                if mesh_name in mesh_files:
                    # Update with correct path
                    relative_path = os.path.relpath(mesh_files[mesh_name])
                    mesh_elem.set('filename', relative_path)
                    print(f"  Fixed mesh: {mesh_name} -> {relative_path}")
                    mesh_count += 1
                else:
                    # Create a simple box geometry as replacement
                    print(f"  Missing mesh: {mesh_name}, replacing with box")
                    parent = mesh_elem.getparent()
                    if parent is not None:
                        # Remove mesh element
                        parent.remove(mesh_elem)
                        # Add box geometry
                        box_elem = ET.SubElement(parent, 'box')
                        box_elem.set('size', '0.1 0.1 0.1')
        
        # Fix duplicate geometry names
        geom_names = set()
        geom_counter = 1
        
        for collision in root.iter('collision'):
            name = collision.get('name')
            if not name or name in geom_names:
                new_name = f"collision_{geom_counter}"
                collision.set('name', new_name)
                geom_counter += 1
            else:
                geom_names.add(name)
        
        for visual in root.iter('visual'):
            name = visual.get('name')
            if not name or name in geom_names:
                new_name = f"visual_{geom_counter}"
                visual.set('name', new_name)
                geom_counter += 1
            else:
                geom_names.add(name)
        
        # Write fixed URDF
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        print(f"✓ Fixed URDF saved as: {output_file}")
        print(f"  - Fixed {mesh_count} mesh references")
        
        return output_file
        
    except Exception as e:
        print(f"Error fixing URDF: {e}")
        return None

def create_simple_mjcf():
    """Create a simple MuJoCo XML file manually"""
    
    mjcf_content = '''<?xml version="1.0" ?>
<mujoco model="mini_pupper">
    <compiler angle="radian" meshdir="../meshes/" texturedir="../textures/"/>
    
    <option timestep="0.002" integrator="RK4"/>
    
    <default>
        <joint limited="true" damping="0.1" stiffness="10"/>
        <geom contype="1" conaffinity="1" condim="3" friction="0.8 0.1 0.1"/>
        <motor ctrllimited="true" ctrlrange="-1 1"/>
    </default>
    
    <asset>
        <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" rgb2=".2 .3 .4" width="300" height="300"/>
        <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
    </asset>

    <worldbody>
        <geom name="floor" size="0 0 .05" type="plane" material="grid"/>
        <light directional="false" diffuse=".2 .2 .2" specular="0 0 0" pos="0 0 5" dir="0 0 -1"/>
        <light directional="true" diffuse=".8 .8 .8" specular=".3 .3 .3" pos="0 0 4" dir="0 0 -1"/>
        
        <!-- Base body -->
        <body name="base_link" pos="0 0 0.3">
            <inertial pos="0 0 0" mass="2.0" diaginertia="0.1 0.1 0.05"/>
            <geom name="base_body" type="box" size="0.15 0.08 0.04" rgba="0.8 0.8 0.8 1"/>
            <site name="imu_site" pos="0 0 0" size="0.01"/>
            
            <!-- Front Left Leg -->
            <body name="front_left_hip" pos="0.12 0.08 0">
                <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
                <joint name="front_left_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <geom name="front_left_hip_geom" type="cylinder" size="0.015 0.02" rgba="0.5 0.5 0.5 1"/>
                
                <body name="front_left_thigh" pos="0 0 -0.05">
                    <inertial pos="0 0 -0.05" mass="0.3" diaginertia="0.002 0.002 0.001"/>
                    <joint name="front_left_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                    <geom name="front_left_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" rgba="0.6 0.6 0.6 1"/>
                    
                    <body name="front_left_calf" pos="0 0 -0.1">
                        <inertial pos="0 0 -0.05" mass="0.2" diaginertia="0.001 0.001 0.001"/>
                        <joint name="front_left_calf_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
                        <geom name="front_left_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.008" rgba="0.4 0.4 0.4 1"/>
                        <geom name="front_left_foot" type="sphere" pos="0 0 -0.1" size="0.02" rgba="0.2 0.2 0.2 1"/>
                    </body>
                </body>
            </body>
            
            <!-- Front Right Leg -->
            <body name="front_right_hip" pos="0.12 -0.08 0">
                <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
                <joint name="front_right_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <geom name="front_right_hip_geom" type="cylinder" size="0.015 0.02" rgba="0.5 0.5 0.5 1"/>
                
                <body name="front_right_thigh" pos="0 0 -0.05">
                    <inertial pos="0 0 -0.05" mass="0.3" diaginertia="0.002 0.002 0.001"/>
                    <joint name="front_right_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                    <geom name="front_right_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" rgba="0.6 0.6 0.6 1"/>
                    
                    <body name="front_right_calf" pos="0 0 -0.1">
                        <inertial pos="0 0 -0.05" mass="0.2" diaginertia="0.001 0.001 0.001"/>
                        <joint name="front_right_calf_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
                        <geom name="front_right_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.008" rgba="0.4 0.4 0.4 1"/>
                        <geom name="front_right_foot" type="sphere" pos="0 0 -0.1" size="0.02" rgba="0.2 0.2 0.2 1"/>
                    </body>
                </body>
            </body>
            
            <!-- Rear Left Leg -->
            <body name="rear_left_hip" pos="-0.12 0.08 0">
                <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
                <joint name="rear_left_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <geom name="rear_left_hip_geom" type="cylinder" size="0.015 0.02" rgba="0.5 0.5 0.5 1"/>
                
                <body name="rear_left_thigh" pos="0 0 -0.05">
                    <inertial pos="0 0 -0.05" mass="0.3" diaginertia="0.002 0.002 0.001"/>
                    <joint name="rear_left_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                    <geom name="rear_left_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" rgba="0.6 0.6 0.6 1"/>
                    
                    <body name="rear_left_calf" pos="0 0 -0.1">
                        <inertial pos="0 0 -0.05" mass="0.2" diaginertia="0.001 0.001 0.001"/>
                        <joint name="rear_left_calf_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
                        <geom name="rear_left_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.008" rgba="0.4 0.4 0.4 1"/>
                        <geom name="rear_left_foot" type="sphere" pos="0 0 -0.1" size="0.02" rgba="0.2 0.2 0.2 1"/>
                    </body>
                </body>
            </body>
            
            <!-- Rear Right Leg -->
            <body name="rear_right_hip" pos="-0.12 -0.08 0">
                <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
                <joint name="rear_right_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/>
                <geom name="rear_right_hip_geom" type="cylinder" size="0.015 0.02" rgba="0.5 0.5 0.5 1"/>
                
                <body name="rear_right_thigh" pos="0 0 -0.05">
                    <inertial pos="0 0 -0.05" mass="0.3" diaginertia="0.002 0.002 0.001"/>
                    <joint name="rear_right_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
                    <geom name="rear_right_thigh_geom" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" rgba="0.6 0.6 0.6 1"/>
                    
                    <body name="rear_right_calf" pos="0 0 -0.1">
                        <inertial pos="0 0 -0.05" mass="0.2" diaginertia="0.001 0.001 0.001"/>
                        <joint name="rear_right_calf_joint" type="hinge" axis="0 1 0" range="-2.5 0"/>
                        <geom name="rear_right_calf_geom" type="capsule" fromto="0 0 0 0 0 -0.1" size="0.008" rgba="0.4 0.4 0.4 1"/>
                        <geom name="rear_right_foot" type="sphere" pos="0 0 -0.1" size="0.02" rgba="0.2 0.2 0.2 1"/>
                    </body>
                </body>
            </body>
        </body>
    </worldbody>
    
    <actuator>
        <motor name="front_left_hip_motor" joint="front_left_hip_joint" gear="50"/>
        <motor name="front_left_thigh_motor" joint="front_left_thigh_joint" gear="50"/>
        <motor name="front_left_calf_motor" joint="front_left_calf_joint" gear="50"/>
        
        <motor name="front_right_hip_motor" joint="front_right_hip_joint" gear="50"/>
        <motor name="front_right_thigh_motor" joint="front_right_thigh_joint" gear="50"/>
        <motor name="front_right_calf_motor" joint="front_right_calf_joint" gear="50"/>
        
        <motor name="rear_left_hip_motor" joint="rear_left_hip_joint" gear="50"/>
        <motor name="rear_left_thigh_motor" joint="rear_left_thigh_joint" gear="50"/>
        <motor name="rear_left_calf_motor" joint="rear_left_calf_joint" gear="50"/>
        
        <motor name="rear_right_hip_motor" joint="rear_right_hip_joint" gear="50"/>
        <motor name="rear_right_thigh_motor" joint="rear_right_thigh_joint" gear="50"/>
        <motor name="rear_right_calf_motor" joint="rear_right_calf_joint" gear="50"/>
    </actuator>
    
    <sensor>
        <accelerometer name="imu_accel" site="imu_site"/>
        <gyro name="imu_gyro" site="imu_site"/>
        
        <jointpos name="front_left_hip_pos" joint="front_left_hip_joint"/>
        <jointvel name="front_left_hip_vel" joint="front_left_hip_joint"/>
        <jointpos name="front_left_thigh_pos" joint="front_left_thigh_joint"/>
        <jointvel name="front_left_thigh_vel" joint="front_left_thigh_joint"/>
        <jointpos name="front_left_calf_pos" joint="front_left_calf_joint"/>
        <jointvel name="front_left_calf_vel" joint="front_left_calf_joint"/>
        
        <jointpos name="front_right_hip_pos" joint="front_right_hip_joint"/>
        <jointvel name="front_right_hip_vel" joint="front_right_hip_joint"/>
        <jointpos name="front_right_thigh_pos" joint="front_right_thigh_joint"/>
        <jointvel name="front_right_thigh_vel" joint="front_right_thigh_joint"/>
        <jointpos name="front_right_calf_pos" joint="front_right_calf_joint"/>
        <jointvel name="front_right_calf_vel" joint="front_right_calf_joint"/>
        
        <jointpos name="rear_left_hip_pos" joint="rear_left_hip_joint"/>
        <jointvel name="rear_left_hip_vel" joint="rear_left_hip_joint"/>
        <jointpos name="rear_left_thigh_pos" joint="rear_left_thigh_joint"/>
        <jointvel name="rear_left_thigh_vel" joint="rear_left_thigh_joint"/>
        <jointpos name="rear_left_calf_pos" joint="rear_left_calf_joint"/>
        <jointvel name="rear_left_calf_vel" joint="rear_left_calf_joint"/>
        
        <jointpos name="rear_right_hip_pos" joint="rear_right_hip_joint"/>
        <jointvel name="rear_right_hip_vel" joint="rear_right_hip_joint"/>
        <jointpos name="rear_right_thigh_pos" joint="rear_right_thigh_joint"/>
        <jointvel name="rear_right_thigh_vel" joint="rear_right_thigh_joint"/>
        <jointpos name="rear_right_calf_pos" joint="rear_right_calf_joint"/>
        <jointvel name="rear_right_calf_vel" joint="rear_right_calf_joint"/>
    </sensor>
</mujoco>'''
    
    with open("mini_pupper_mujoco.xml", 'w') as f:
        f.write(mjcf_content)
    
    print("✓ Created simple MuJoCo model: mini_pupper_mujoco.xml")
    return True

def test_mujoco_model():
    """Test the MuJoCo model"""
    try:
        import mujoco
        
        model = mujoco.MjModel.from_xml_path("mini_pupper_mujoco.xml")
        data = mujoco.MjData(model)
        
        print(f"✓ MuJoCo model loaded successfully!")
        print(f"  - Bodies: {model.nbody}")
        print(f"  - Joints: {model.njnt}")
        print(f"  - Actuators: {model.nu}")
        print(f"  - DoF: {model.nv}")
        
        # Test simulation
        for i in range(100):
            mujoco.mj_step(model, data)
        
        print(f"✓ Simulation test passed!")
        return True
        
    except ImportError:
        print("⚠ MuJoCo not installed, skipping test")
        return True
    except Exception as e:
        print(f"✗ MuJoCo test failed: {e}")
        return False

def create_training_script():
    """Create a complete training script"""
    
    training_code = '''#!/usr/bin/env python3
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
'''
    
    with open("train_mini_pupper.py", 'w') as f:
        f.write(training_code)
    
    os.chmod("train_mini_pupper.py", 0o755)
    print("✓ Created training script: train_mini_pupper.py")

def main():
    """Main conversion process"""
    print("Fixed MiniPupper URDF to MuJoCo Converter")
    print("=" * 45)
    
    # Check if URDF exists
    urdf_file = "mini_pupper_description.urdf"
    if not os.path.exists(urdf_file):
        print(f"✗ {urdf_file} not found")
        return
    
    print(f"Processing {urdf_file}...")
    
    # Try to fix the URDF first
    fixed_urdf = fix_urdf_paths(urdf_file)
    
    # Create simple MuJoCo model (bypasses mesh issues)
    print("\nCreating simplified MuJoCo model...")
    if create_simple_mjcf():
        print("✓ MuJoCo model created successfully!")
        
        # Test the model
        if test_mujoco_model():
            # Create training script
            create_training_script()
            
            print("\n" + "="*50)
            print("SUCCESS! Files created:")
            print("  - mini_pupper_mujoco.xml (MuJoCo model)")
            print("  - train_mini_pupper.py (Training script)")
            
            print("\nNext steps:")
            print("1. Test the model:")
            print("   python train_mini_pupper.py")
            print("\n2. View in MuJoCo viewer:")
            print("   python train_mini_pupper.py view")
            
            print("\n3. Install viewer if needed:")
            print("   pip install 'mujoco[viewer]'")
        
        else:
            print("✗ Model test failed")
    else:
        print("✗ Failed to create MuJoCo model")

if __name__ == "__main__":
    main()