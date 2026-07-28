import numpy as np
import pandas as pd
import time
import os
from mlp_controller_v3 import FixedMappingControllerV3


def analyze_joint_actions(csv_path):
    """
    Analyze the actions in a CSV to get average action per joint.

    Returns:
        dict: Average action value for each joint
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return None

    try:
        df = pd.read_csv(csv_path)

        joint_names = [
            'action_base_lf1', 'action_lf1_lf2', 'action_lf2_lf3',  # LF
            'action_base_rf1', 'action_rf1_rf2', 'action_rf2_rf3',  # RF
            'action_base_lb1', 'action_lb1_lb2', 'action_lb2_lb3',  # LB
            'action_base_rb1', 'action_rb1_rb2', 'action_rb2_rb3',  # RB
        ]

        # Calculate average action for each joint (skip first 10 steps)
        start_idx = 10
        avg_actions = {}

        for joint_name in joint_names:
            if joint_name in df.columns:
                avg_actions[joint_name] = df[joint_name].iloc[start_idx:].mean()
            else:
                print(f"[WARNING] Column {joint_name} not found")
                avg_actions[joint_name] = 0.0

        return avg_actions

    except Exception as e:
        print(f"[ERROR] Could not analyze CSV: {e}")
        return None


def test_single_joint(controller, joint_idx, action_value, duration=2.0, test_name=""):

    print(f"\n{'='*70}")
    print(f"TESTING JOINT {joint_idx}: {test_name}")
    print(f"CSV Action Value: {action_value:+.4f}")
    print(f"{'='*70}")

    # Parse joint info
    leg_name = test_name.split('_')[0]  # LF, RF, LB, RB
    joint_type = test_name.split('_')[1]  # hip, thigh, calf

    print(f"\n[WATCH] Focus on the {leg_name} leg, {joint_type} joint")
    print(f"[EXPECT] This joint should move when action is applied")

    # Describe expected movement based on joint type
    if 'hip' in joint_type.lower():
        print(f"[EXPECT] Hip joint - should swing leg forward/backward or left/right")
    elif 'thigh' in joint_type.lower():
        print(f"[EXPECT] Thigh joint - should lift/lower the leg")
    elif 'calf' in joint_type.lower():
        print(f"[EXPECT] Calf joint - should extend/retract lower leg")

    if action_value > 0:
        print(f"[EXPECT] Positive action ({action_value:+.4f}) - one direction")
    else:
        print(f"[EXPECT] Negative action ({action_value:+.4f}) - opposite direction")

    print(f"\n[INFO] Joint will move in {duration:.1f} seconds")
    print(f"[INFO] Starting in 2 seconds... WATCH CLOSELY!")
    time.sleep(2)

    # Create action array with only this joint active
    actions = np.zeros(12)
    actions[joint_idx] = action_value

    # Apply action repeatedly for duration
    steps = int(duration * 50)  # 50Hz control rate
    start_time = time.time()

    for i in range(steps):
        # Use simulation ACTION_SCALE = 0.5 (same as in CSV replay)
        target_sim = controller.sim_default_positions + actions * 0.5
        target_sim = np.clip(target_sim, controller.joint_lower_limits,
                            controller.joint_upper_limits)

        target_matrix = controller._isaac_to_hardware_matrix(target_sim)
        controller.hardware.set_actuator_postions(target_matrix)

        time.sleep(0.02)  # 50Hz

    total_time = time.time() - start_time
    print(f"\n[DONE] Motion completed in {total_time:.1f}s")

    # Keep position held for observation
    print(f"[HOLD] Holding position for 1 second to observe...")
    time.sleep(1)

    # Return to default position
    print(f"[INFO] Returning to default stance...")
    target_matrix = controller._isaac_to_hardware_matrix(controller.sim_default_positions)
    controller.hardware.set_actuator_postions(target_matrix)
    time.sleep(1.0)


def main():
    print("="*70)
    print("INDIVIDUAL JOINT ACTION TEST")
    print("="*70)
    print("""
This script tests each joint individually to verify transformations.

We'll use actions from a FORWARD MOTION CSV (where sim showed forward movement).
Each joint will be tested one at a time.

For each joint, observe:
  - Does the leg move as expected?
  - Does it cause forward/backward motion?
  - Does it cause left/right motion?
  - Does it cause rotation?
  - Does nothing happen?

This will help identify which joints have incorrect transformations.
""")

    # Select CSV to analyze (use forward motion that worked in sim)
    csv_path = "/home/ubuntu/debug/obs_action_logs_x_010/env_1_actions.csv"

    print(f"\n[INFO] Analyzing actions from: {csv_path}")
    avg_actions = analyze_joint_actions(csv_path)

    if avg_actions is None:
        print("[ERROR] Could not load CSV. Exiting.")
        return

    # Display average actions
    joint_labels = [
        'LF_hip', 'LF_thigh', 'LF_calf',  # Left Front
        'RF_hip', 'RF_thigh', 'RF_calf',  # Right Front
        'LB_hip', 'LB_thigh', 'LB_calf',  # Left Back
        'RB_hip', 'RB_thigh', 'RB_calf',  # Right Back
    ]

    joint_action_pairs = []

    print(f"\n{'='*70}")
    print("AVERAGE JOINT ACTIONS FROM FORWARD MOTION CSV")
    print(f"{'='*70}")
    for i, (joint_name, joint_label) in enumerate(zip(avg_actions.keys(), joint_labels)):
        action_val = avg_actions[joint_name]
        print(f"  Joint {i:2d} ({joint_label:12s}): {action_val:+.4f}")
        joint_action_pairs.append((i, joint_label, action_val))

    print(f"\n[INFO] This CSV showed FORWARD motion in simulation")
    print(f"[INFO] If transformations are correct, joints should contribute to forward motion\n")

    input("Press Enter to begin individual joint tests...")

    # Initialize controller
    print("\n[INFO] Initializing controller...")
    controller = FixedMappingControllerV3("/home/ubuntu/mp2_mlp/policy_joyboy.pt")

    # Test each joint individually
    results = []

    for joint_idx, joint_label, action_val in joint_action_pairs:
        # Skip joints with very small actions
        if abs(action_val) < 0.01:
            print(f"\n[SKIP] Joint {joint_idx} ({joint_label}) has negligible action ({action_val:+.4f})")
            results.append({
                'joint_idx': joint_idx,
                'joint_label': joint_label,
                'action_val': action_val,
                'observation': 'SKIPPED - negligible action'
            })
            continue

        # Test this joint
        test_single_joint(controller, joint_idx, action_val,
                         duration=3.0, test_name=joint_label)

        # Ask user what they observed - focus on JOINT-LEVEL correctness
        print(f"\n[OBSERVATION] What happened when you applied action to {joint_label}?")
        print("\nPart 1: Which PHYSICAL JOINT moved?")
        print("  1. CORRECT joint moved (the one we expected)")
        print("  2. WRONG joint moved (different leg or joint)")
        print("  3. Multiple joints moved")
        print("  4. No joints moved")

        which_joint = input("Enter number (1-4): ").strip()

        print("\nPart 2: Did it move in the CORRECT DIRECTION?")
        print("  (Only answer if correct joint moved)")
        print("  1. YES - Moved in expected direction")
        print("  2. NO - Moved in OPPOSITE direction (sign flip)")
        print("  3. Unclear / Not applicable")

        direction = input("Enter number (1-3): ").strip()

        # Interpret results
        joint_map = {
            "1": "CORRECT joint",
            "2": "WRONG joint",
            "3": "MULTIPLE joints",
            "4": "NO movement"
        }
        dir_map = {
            "1": "CORRECT direction",
            "2": "OPPOSITE direction (SIGN FLIP)",
            "3": "Unclear"
        }

        joint_obs = joint_map.get(which_joint, "Unknown")
        dir_obs = dir_map.get(direction, "Unknown")

        # Determine overall status
        if which_joint == "1" and direction == "1":
            status = "CORRECT"
            observation = f"{joint_obs} + {dir_obs}"
        elif which_joint == "1" and direction == "2":
            status = "SIGN FLIP"
            observation = f"{joint_obs} + {dir_obs}"
        elif which_joint == "2":
            status = "WRONG JOINT MAPPING"
            observation = joint_obs
        elif which_joint == "3":
            status = "MULTIPLE JOINTS (COUPLING ERROR)"
            observation = joint_obs
        elif which_joint == "4":
            status = "NO MOVEMENT"
            observation = joint_obs
        else:
            status = "NCLEAR"
            observation = f"{joint_obs} + {dir_obs}"

        print(f"\n[RESULT] {status}: {observation}")

        results.append({
            'joint_idx': joint_idx,
            'joint_label': joint_label,
            'action_val': action_val,
            'status': status,
            'observation': observation
        })

        # Ask if user wants to continue
        if joint_idx < 11:
            cont = input(f"\nPress Enter to test next joint, or 'q' to quit: ").strip().lower()
            if cont == 'q':
                break

    # Summary
    print("\n" + "="*70)
    print("JOINT-LEVEL TRANSFORMATION TEST SUMMARY")
    print("="*70)

    correct = []
    sign_flip = []
    wrong_joint = []
    no_movement = []
    other = []

    for result in results:
        if 'status' not in result:
            # Handle skipped results
            other.append(result)
            continue

        status_str = f"Joint {result['joint_idx']:2d} ({result['joint_label']:12s}): " \
                    f"action={result['action_val']:+.4f}"

        if "✅" in result['status']:
            correct.append(result)
            print(f"   {status_str} → {result['observation']}")
        elif "SIGN FLIP" in result['status']:
            sign_flip.append(result)
            print(f"    {status_str} → {result['observation']}")
        elif "WRONG JOINT" in result['status']:
            wrong_joint.append(result)
            print(f"   {status_str} → {result['observation']}")
        elif "NO MOVEMENT" in result['status']:
            no_movement.append(result)
            print(f"   {status_str} → {result['observation']}")
        else:
            other.append(result)
            print(f"   {status_str} → {result['observation']}")

    print(f"\n{'='*70}")
    print("DIAGNOSTIC ANALYSIS")
    print(f"{'='*70}")
    print(f"   Correct (joint + direction):     {len(correct)}/12")
    print(f"   Sign flips (joint OK, dir wrong): {len(sign_flip)}/12")
    print(f"   Wrong joint mapping:              {len(wrong_joint)}/12")
    print(f"   No movement:                      {len(no_movement)}/12")
    print(f"   Other/Unclear:                    {len(other)}/12")

    # Detailed problem analysis
    if sign_flip:
        print(f"\n{'='*70}")
        print("SIGN FLIP ISSUES - Fix joint_direction array:")
        print(f"{'='*70}")
        for result in sign_flip:
            print(f"  Joint {result['joint_idx']:2d} ({result['joint_label']:12s}): Flip sign in joint_direction")

    if wrong_joint:
        print(f"\n{'='*70}")
        print("JOINT MAPPING ISSUES - Fix _isaac_to_hardware_matrix:")
        print(f"{'='*70}")
        for result in wrong_joint:
            print(f"  Joint {result['joint_idx']:2d} ({result['joint_label']:12s}): Check leg/joint mapping")

    if no_movement:
        print(f"\n{'='*70}")
        print("NO MOVEMENT ISSUES:")
        print(f"{'='*70}")
        for result in no_movement:
            print(f"  Joint {result['joint_idx']:2d} ({result['joint_label']:12s}): Check transformation pipeline")

    # Overall assessment
    print(f"\n{'='*70}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*70}")

    total_working = len(correct)
    total_fixable = len(sign_flip)
    total_broken = len(wrong_joint) + len(no_movement)

    if total_working == 12:
        print(f"  PERFECT - All transformations are correct!")
    elif total_working + total_fixable == 12:
        print(f"   FIXABLE - All joints map correctly, just need sign flips")
        print(f"     → Update joint_direction array in mlp_controller_v3.py")
    elif total_broken > 0:
        print(f"   BROKEN - {total_broken} joints have mapping errors")
        print(f"     → Need to fix _isaac_to_hardware_matrix() or hw_to_isaac_leg mapping")
    else:
        print(f"   UNCLEAR - Need more investigation")

    print(f"\n{'='*70}")
    print("[DONE] Individual joint testing complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test cancelled")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
