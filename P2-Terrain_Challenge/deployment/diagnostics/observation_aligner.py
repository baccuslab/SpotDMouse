import numpy as np
import torch
from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import time

class IsaacSimObservationAligner:
    """
    Aligns real robot observations with Isaac Sim training format.
    Based on SpotObservationsCfg configuration.
    """
    
    def __init__(self):
        self.esp32 = ESP32Interface()
        time.sleep(0.5)
        
        # Servo calibration
        self.servo_offset = 512 
        self.servo_scale = 1024 / (2 * np.pi)
        
        #training order in sim
        self.isaac_joint_order = [
            "base_lf1", "lf1_lf2", "lf2_lf3",  # LF leg
            "base_rf1", "rf1_rf2", "rf2_rf3",  # RF leg  
            "base_lb1", "lb1_lb2", "lb2_lb3",  # LB leg
            "base_rb1", "rb1_rb2", "rb2_rb3"   # RB leg
        ]

        #executiuon order on device
        self.esp32_servo_order = [
            3, 4, 5,    # LF -> FL: servos 3,4,5
            0, 1, 2,    # RF -> FR: servos 0,1,2
            9, 10, 11,  # LB -> BL: servos 9,10,11
            6, 7, 8     # RB -> BR: servos 6,7,8
        ]
        
        # Get default positions from Isaac Sim config
        # These are the standing positions your policy expects
        self.isaac_default_positions = np.array([
            0.0, 0.8, -1.57,  # LF (from simulation)
             0.0, 0.8, -1.57,  # RF
            0.0, 0.8, -1.57,  # LB
             0.0, 0.8, -1.57   # RB
        ])
        
        # Get current robot positions
        current_servo_positions = np.array(self.esp32.servos_get_position())
        self.current_positions_rad = (current_servo_positions - self.servo_offset) / self.servo_scale
        
        # State tracking
        self.prev_positions = self.current_positions_rad.copy()
        self.prev_actions = np.zeros(12)
        self.prev_time = time.time()
        
        # Velocity estimation with smoothing
        self.velocity_alpha = 0.3  # Exponential smoothing factor
        self.estimated_base_vel = np.zeros(3)
        
        # Noise parameters from training config
        self.noise_params = {
            'base_lin_vel': (-0.1, 0.1),
            'base_ang_vel': (-0.2, 0.2),
            'projected_gravity': (-0.05, 0.05),
            'joint_pos': (-0.01, 0.01),
            'joint_vel': (-1.5, 1.5),
            'joint_effort': (-0.1, 0.1)
        }
        
        print("Isaac Sim Observation Aligner initialized")
        print(f"Current positions (rad): {self.current_positions_rad.round(3)}")
        print(f"Isaac default positions: {self.isaac_default_positions.round(3)}")
        print(f"Position difference: {(self.current_positions_rad - self.isaac_default_positions).round(3)}")
        
    def calibrate_to_isaac_defaults(self):
        """Move robot to Isaac Sim default standing position"""
        print("\nCalibrating to Isaac Sim default position...")
        
        target_servos = (self.isaac_default_positions * self.servo_scale + self.servo_offset)
        target_servos = np.clip(target_servos, 100, 924)
        servo_positions = [int(pos) for pos in target_servos]
        
        # Move gradually to avoid jerky motion
        steps = 20
        current_servos = self.esp32.servos_get_position()
        
        for i in range(steps):
            alpha = (i + 1) / steps
            interpolated = current_servos + alpha * (np.array(servo_positions) - current_servos)
            interpolated_int = [int(pos) for pos in interpolated]
            self.esp32.servos_set_position_torque(interpolated_int, [1]*12)
            time.sleep(0.05)
        
        print("Calibration complete!")
        
        # Update current positions
        current_servo_positions = np.array(self.esp32.servos_get_position())
        self.current_positions_rad = (current_servo_positions - self.servo_offset) / self.servo_scale
        self.prev_positions = self.current_positions_rad.copy()
        
    def get_isaac_aligned_observation(self, velocity_command):
        """
        Get observation in exact Isaac Sim format.
        
        Isaac Sim observation structure (60 dims total):
        - base_lin_vel: 3 dims
        - base_ang_vel: 3 dims  
        - projected_gravity: 3 dims
        - velocity_commands: 3 dims
        - joint_pos (relative): 12 dims
        - joint_vel: 12 dims
        - joint_effort: 12 dims
        - actions (previous): 12 dims
        """
        
        current_time = time.time()
        dt = current_time - self.prev_time
        
        # Get sensor data
        raw_positions = np.array(self.esp32.servos_get_position())
        raw_loads = np.array(self.esp32.servos_get_load())
        imu_data = self.esp32.imu_get_data()
        
        # 1. Joint positions (relative to Isaac defaults)
        current_positions_rad = (raw_positions - self.servo_offset) / self.servo_scale
        joint_pos_relative = current_positions_rad - self.isaac_default_positions
        
        # 2. Joint velocities
        if dt > 0 and dt < 0.1:
            joint_velocities = (current_positions_rad - self.prev_positions) / dt
            # Apply same clipping as training
            joint_velocities = np.clip(joint_velocities, -50, 50)
        else:
            joint_velocities = np.zeros(12)
        
        # 3. Joint efforts (normalized like in training)
        # Isaac Sim uses applied_torque, we approximate with servo loads
        joint_efforts = raw_loads / 500.0  # Normalize to roughly [-1, 1]
        
        # 4. Base angular velocity (rad/s)
        base_ang_vel = np.array([
            imu_data['gx'] * 0.01745,  # deg/s to rad/s
            imu_data['gy'] * 0.01745,
            imu_data['gz'] * 0.01745
        ])
        
        # 5. Projected gravity (normalized accelerometer)
        acc_mag = np.sqrt(imu_data['ax']**2 + imu_data['ay']**2 + imu_data['az']**2)
        if acc_mag > 0.1:
            projected_gravity = np.array([
                imu_data['ax'] / acc_mag,
                imu_data['ay'] / acc_mag,
                imu_data['az'] / acc_mag
            ])
        else:
            projected_gravity = np.array([0.0, 0.0, -1.0])
        
        # 6. Base linear velocity estimation
        # This is the trickiest part - Isaac Sim has perfect odometry
        # We need to estimate it from commanded velocity and joint motion
        base_lin_vel = self.estimate_base_velocity(
            velocity_command, 
            joint_velocities, 
            base_ang_vel, 
            dt
        )
        
        # Build observation vector (must be exactly 60 dims)
        observation = np.concatenate([
            base_lin_vel,           # 3
            base_ang_vel,          # 3
            projected_gravity,     # 3
            velocity_command,      # 3
            joint_pos_relative,    # 12
            joint_velocities,      # 12
            joint_efforts,         # 12
            self.prev_actions      # 12
        ])
        
        # Sanity check
        assert observation.shape[0] == 60, f"Wrong observation size: {observation.shape[0]}"
        
        # Update state for next iteration
        self.prev_positions = current_positions_rad.copy()
        self.prev_time = current_time
        
        return observation
    
    def estimate_base_velocity(self, command, joint_vel, ang_vel, dt):
        """
        Estimate base velocity using multiple signals.
        Isaac Sim has perfect odometry, we need to approximate.
        """
        
        # Method 1: Command tracking with delay
        # Robot velocity lags behind command
        command_contribution = command * 0.7  # Assume 70% tracking
        
        # Method 2: Use joint velocities as motion indicator
        # Higher joint velocity = more likely to be moving
        avg_joint_motion = np.mean(np.abs(joint_vel))
        motion_scale = np.tanh(avg_joint_motion / 5.0)  # Saturates around 5 rad/s
        
        # Method 3: IMU-based dead reckoning (simplified)
        # For now, just use angular velocity to detect turning
        turning_indicator = abs(ang_vel[2])  # Yaw rate
        
        # Combine estimates
        estimated_vx = command[0] * motion_scale
        estimated_vy = command[1] * motion_scale * 0.5  # Lateral is harder
        estimated_vz = 0.0  # Assume flat ground
        
        # Smooth the estimate (exponential moving average)
        new_estimate = np.array([estimated_vx, estimated_vy, estimated_vz])
        self.estimated_base_vel = (self.velocity_alpha * new_estimate + 
                                  (1 - self.velocity_alpha) * self.estimated_base_vel)
        
        return self.estimated_base_vel
    
    def validate_observation_ranges(self, obs):
        """Check if observation is within training ranges"""
        
        print("\nValidating observation against training ranges:")
        
        # Expected ranges from training (with noise)
        checks = [
            ("base_lin_vel", 0, 3, -2.0, 2.0),
            ("base_ang_vel", 3, 6, -4.0, 4.0),
            ("projected_gravity", 6, 9, -1.1, 1.1),
            ("velocity_commands", 9, 12, -1.0, 1.0),
            ("joint_pos", 12, 24, -3.14, 3.14),
            ("joint_vel", 24, 36, -50.0, 50.0),
            ("joint_effort", 36, 48, -10.0, 10.0),
            ("actions", 48, 60, -2.0, 2.0)
        ]
        
        all_valid = True
        for name, start, end, min_val, max_val in checks:
            component = obs[start:end]
            min_actual = np.min(component)
            max_actual = np.max(component)
            
            in_range = min_actual >= min_val and max_actual <= max_val
            status = "✓" if in_range else "✗"
            
            print(f"{status} {name:20s}: [{min_actual:6.2f}, {max_actual:6.2f}] "
                  f"(expected [{min_val:6.2f}, {max_val:6.2f}])")
            
            if not in_range:
                all_valid = False
        
        return all_valid
    
    def test_policy_with_isaac_obs(self, policy_path="/home/ubuntu/mp2_mlp/policy_only.pt"):
        """Test policy with properly aligned observations"""
        
        print("\nTesting policy with Isaac-aligned observations...")
        
        # Load policy
        policy = torch.jit.load(policy_path)
        policy.eval()
        
        # Test different velocity commands
        test_commands = [
            ([0.0, 0.0, 0.0], "Standing"),
            ([0.3, 0.0, 0.0], "Forward slow"),
            ([0.5, 0.0, 0.0], "Forward medium"),
            ([0.0, 0.0, 0.3], "Turn right"),
            ([-0.3, 0.0, 0.0], "Backward")
        ]
        
        for command, description in test_commands:
            print(f"\n{description} (command={command}):")
            
            # Get Isaac-aligned observation
            obs = self.get_isaac_aligned_observation(np.array(command))
            
            # Validate ranges
            valid = self.validate_observation_ranges(obs)
            
            # Run through policy
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                actions = policy(obs_tensor).squeeze().numpy()
            
            print(f"Actions: mean={np.mean(actions):.3f}, "
                  f"std={np.std(actions):.3f}, "
                  f"range=[{np.min(actions):.3f}, {np.max(actions):.3f}]")
            
            # Update prev_actions for next iteration
            self.prev_actions = actions
            
            time.sleep(0.5)

def main():
    print("="*60)
    print("ISAAC SIM OBSERVATION ALIGNMENT TOOL")
    print("="*60)
    
    aligner = IsaacSimObservationAligner()
    
    print("\nOptions:")
    print("1. Calibrate robot to Isaac Sim default position")
    print("2. Test observation alignment")
    print("3. Run full diagnostic with policy")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        aligner.calibrate_to_isaac_defaults()
    elif choice == "2":
        # Test single observation
        command = np.array([0.5, 0.0, 0.0])  # Forward
        obs = aligner.get_isaac_aligned_observation(command)
        aligner.validate_observation_ranges(obs)
    elif choice == "3":
        aligner.test_policy_with_isaac_obs()

if __name__ == "__main__":
    main()