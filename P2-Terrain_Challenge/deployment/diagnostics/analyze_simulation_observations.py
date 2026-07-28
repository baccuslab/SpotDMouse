import pandas as pd
import numpy as np
import os
from pathlib import Path

def analyze_observation_file(csv_path):
    """
    Analyze a single observation CSV file.

    Returns average velocities: (vx, vy, vyaw)
    """
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)

        # Column names from your CSV files
        vx_col = 'base_lin_vel_x'
        vy_col = 'base_lin_vel_y'
        vyaw_col = 'base_ang_vel_z'  # Z-axis rotation (yaw)

        # Verify columns exist
        if vx_col not in df.columns or vy_col not in df.columns or vyaw_col not in df.columns:
            print(f"[ERROR] Missing required columns in {csv_path}")
            print(f"Available columns: {list(df.columns[:10])}")
            return None

        # Calculate average velocities (skip first 10 steps for startup)
        start_idx = min(10, len(df) // 4)
        end_idx = len(df)

        vx_avg = df[vx_col].iloc[start_idx:end_idx].mean()
        vy_avg = df[vy_col].iloc[start_idx:end_idx].mean()
        vyaw_avg = df[vyaw_col].iloc[start_idx:end_idx].mean()

        return {
            'vx': vx_avg,
            'vy': vy_avg,
            'vyaw': vyaw_avg,
            'vx_std': df[vx_col].iloc[start_idx:end_idx].std(),
            'vy_std': df[vy_col].iloc[start_idx:end_idx].std(),
            'vyaw_std': df[vyaw_col].iloc[start_idx:end_idx].std(),
            'num_steps': len(df)
        }

    except Exception as e:
        print(f"[ERROR] Could not analyze {csv_path}: {e}")
        return None


def interpret_velocity(vx, vy, vyaw, threshold=0.01):
    """
    Interpret what movement the velocity represents.

    Returns a string describing the movement.
    """
    movements = []

    # Linear movements
    if abs(vx) > threshold:
        if vx > 0:
            movements.append(f"Forward(vx={vx:+.3f})")
        else:
            movements.append(f"Backward(vx={vx:+.3f})")

    if abs(vy) > threshold:
        if vy > 0:
            movements.append(f"Left(vy={vy:+.3f})")
        else:
            movements.append(f"Right(vy={vy:+.3f})")

    # Angular movement
    if abs(vyaw) > threshold:
        if vyaw > 0:
            movements.append(f"TurnLeft(vyaw={vyaw:+.3f})")
        else:
            movements.append(f"TurnRight(vyaw={vyaw:+.3f})")

    if not movements:
        return "Stationary"

    return " + ".join(movements)


def analyze_all_observations(base_dir="/Users/javierweddington/debug"):
    """
    Analyze all observation files across all directions and environments.
    """

    test_directions = [
        ("obs_action_logs_x_010", "Forward X+0.10"),
        ("obs_action_logs_x_030", "Forward X+0.30"),
        ("obs_action_logs_x_n030", "Backward X-0.30"),
        ("obs_action_logs_y_010", "Strafe Left Y+0.10"),
        ("obs_action_logs_y_030", "Strafe Left Y+0.30"),
        ("obs_action_logs_y_n030", "Strafe Right Y-0.30"),
        ("obs_action_logs_z_010", "Turn Left Z+0.10"),
        ("obs_action_logs_z_030", "Turn Left Z+0.30"),
        ("obs_action_logs_z_n030", "Turn Right Z-0.30"),
    ]

    results = {}

    print("="*80)
    print("SIMULATION OBSERVATION ANALYSIS")
    print("="*80)
    print("\nAnalyzing what the robot ACTUALLY did in simulation...")
    print()

    for dir_name, label in test_directions:
        print(f"\n{'='*80}")
        print(f"DIRECTION: {dir_name} (Labeled as: {label})")
        print(f"{'='*80}")

        direction_results = []

        for env_num in range(6):
            obs_csv = f"{base_dir}/{dir_name}/env_{env_num}_observations.csv"

            print(f"\n  Env {env_num}: ", end="")

            vel_data = analyze_observation_file(obs_csv)

            if vel_data is None:
                print("❌ FILE NOT FOUND or ERROR")
                direction_results.append({
                    'env': env_num,
                    'found': False
                })
                continue

            movement = interpret_velocity(vel_data['vx'], vel_data['vy'], vel_data['vyaw'])

            print(f"{movement}")
            print(f"           vx={vel_data['vx']:+.4f} (±{vel_data['vx_std']:.4f}), "
                  f"vy={vel_data['vy']:+.4f} (±{vel_data['vy_std']:.4f}), "
                  f"vyaw={vel_data['vyaw']:+.4f} (±{vel_data['vyaw_std']:.4f})")

            direction_results.append({
                'env': env_num,
                'found': True,
                'vx': vel_data['vx'],
                'vy': vel_data['vy'],
                'vyaw': vel_data['vyaw'],
                'vx_std': vel_data['vx_std'],
                'vy_std': vel_data['vy_std'],
                'vyaw_std': vel_data['vyaw_std'],
                'movement': movement,
                'num_steps': vel_data['num_steps']
            })

        results[dir_name] = {
            'label': label,
            'envs': direction_results
        }

    return results


def generate_summary_report(results):
    """
    Generate a summary report analyzing consistency and correctness.
    """
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    for dir_name, data in results.items():
        label = data['label']
        envs = [e for e in data['envs'] if e['found']]

        if not envs:
            continue

        print(f"\n{dir_name} (Labeled: {label})")
        print("-" * 80)

        # Calculate average velocities across all environments
        avg_vx = np.mean([e['vx'] for e in envs])
        avg_vy = np.mean([e['vy'] for e in envs])
        avg_vyaw = np.mean([e['vyaw'] for e in envs])

        std_vx = np.std([e['vx'] for e in envs])
        std_vy = np.std([e['vy'] for e in envs])
        std_vyaw = np.std([e['vyaw'] for e in envs])

        print(f"  Average across {len(envs)} envs:")
        print(f"    vx   = {avg_vx:+.4f} (std={std_vx:.4f})")
        print(f"    vy   = {avg_vy:+.4f} (std={std_vy:.4f})")
        print(f"    vyaw = {avg_vyaw:+.4f} (std={std_vyaw:.4f})")

        # Determine dominant movement
        dominant_movement = interpret_velocity(avg_vx, avg_vy, avg_vyaw)
        print(f"  Dominant Movement: {dominant_movement}")

        # Check consistency
        if std_vx > 0.02 or std_vy > 0.02 or std_vyaw > 0.05:
            print(f"  ⚠️  HIGH VARIANCE - Environments are INCONSISTENT")
        else:
            print(f"  ✓  Low variance - Environments are CONSISTENT")

        # Check if label matches actual movement
        label_lower = label.lower()
        movement_lower = dominant_movement.lower()

        match = False
        if 'forward' in label_lower and 'forward' in movement_lower:
            match = True
        elif 'backward' in label_lower and 'backward' in movement_lower:
            match = True
        elif 'left' in label_lower and 'strafe' not in label_lower and 'turn' in label_lower and 'turnleft' in movement_lower:
            match = True
        elif 'right' in label_lower and 'strafe' not in label_lower and 'turn' in label_lower and 'turnright' in movement_lower:
            match = True
        elif 'strafe left' in label_lower and 'left' in movement_lower and 'turn' not in movement_lower:
            match = True
        elif 'strafe right' in label_lower and 'right' in movement_lower and 'turn' not in movement_lower:
            match = True

        if match:
            print(f"  ✅ LABEL MATCHES ACTUAL MOVEMENT")
        else:
            print(f"  ❌ LABEL MISMATCH - Expected: {label}, Got: {dominant_movement}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("""
If labels MATCH actual movements in simulation:
  → Problem is in the REAL ROBOT transformation (mlp_controller.py)
  → The simulation data is correct

If labels DON'T MATCH actual movements in simulation:
  → Problem is in the SIMULATION command application or data labeling
  → Need to fix the simulation environment
  → Or relabel the data with correct movement descriptions
""")


def main():
    # Try both macOS and Ubuntu paths
    if os.path.exists("/Users/javierweddington/debug"):
        base_dir = "/Users/javierweddington/debug"
    elif os.path.exists("/home/ubuntu/debug"):
        base_dir = "/home/ubuntu/debug"
    else:
        print("[ERROR] Could not find debug directory")
        print("Please update the base_dir path in the script")
        return

    print(f"[INFO] Using base directory: {base_dir}\n")

    results = analyze_all_observations(base_dir)
    generate_summary_report(results)

    print("\n[DONE] Analysis complete")


if __name__ == "__main__":
    main()
