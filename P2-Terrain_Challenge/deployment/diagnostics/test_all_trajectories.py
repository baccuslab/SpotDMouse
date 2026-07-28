"""
Replays all simulation action logs to validate joint transformations
across different velocity commands (x, y, z directions, positive and negative)
"""

import numpy as np
import time
import os
from mlp_controller_v3 import FixedMappingControllerV3

def test_trajectory(controller, csv_path, description, max_steps=150):
    """Test a single trajectory from CSV."""
    import pandas as pd

    print("\n" + "="*70)
    print(f"TESTING: {description}")
    print(f"CSV: {csv_path}")
    print("="*70)

    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return False

    try:
        sim_actions = pd.read_csv(csv_path)
        print(f"[INFO] Loaded {len(sim_actions)} steps from CSV")
    except Exception as e:
        print(f"[ERROR] Could not load CSV: {e}")
        return False

    print("[INFO] Starting replay in 3 seconds...")
    print("        Watch the robot's movement direction!")
    time.sleep(3)

    start_time = time.time()

    for i in range(min(max_steps, len(sim_actions))):
        row = sim_actions.iloc[i]
        actions = np.array([
            row['action_base_lf1'], row['action_lf1_lf2'], row['action_lf2_lf3'],
            row['action_base_rf1'], row['action_rf1_rf2'], row['action_rf2_rf3'],
            row['action_base_lb1'], row['action_lb1_lb2'], row['action_lb2_lb3'],
            row['action_base_rb1'], row['action_rb1_rb2'], row['action_rb2_rb3'],
        ])

        # Apply with sim ACTION_SCALE = 0.5
        target_sim = controller.sim_default_positions + actions * 0.5
        target_sim = np.clip(target_sim, controller.joint_lower_limits,
                            controller.joint_upper_limits)

        target_matrix = controller._isaac_to_hardware_matrix(target_sim)
        controller.hardware.set_actuator_postions(target_matrix)

        time.sleep(0.02)  # 50Hz

        if i % 25 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {i:3d}/{max_steps}: "
                  f"actions=[{actions.min():+.2f}, {actions.max():+.2f}] "
                  f"t={elapsed:.1f}s")

    total_time = time.time() - start_time
    print(f"[DONE] Completed {max_steps} steps in {total_time:.1f}s")
    print(f"[INFO] Robot should have moved in expected direction")
    print(f"[INFO] Returning to default stance...")
    time.sleep(2)

    # Return to default
    target_matrix = controller._isaac_to_hardware_matrix(controller.sim_default_positions)
    controller.hardware.set_actuator_postions(target_matrix)
    time.sleep(1)

    return True


def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE SIMULATION TRAJECTORY TEST SUITE")
    print("="*70)
    print("""
This script will test ALL 6 environments for each direction to empirically
discover which simulation logs actually produce the correct real-world movement.

Directions to test:
  1. obs_action_logs_x_010   (labeled: forward, vx=+0.10)
  2. obs_action_logs_x_030   (labeled: forward, vx=+0.30)
  3. obs_action_logs_x_n030  (labeled: backward, vx=-0.30)
  4. obs_action_logs_y_010   (labeled: strafe left, vy=+0.10)
  5. obs_action_logs_y_030   (labeled: strafe left, vy=+0.30)
  6. obs_action_logs_y_n030  (labeled: strafe right, vy=-0.30)
  7. obs_action_logs_z_010   (labeled: turn left, vyaw=+0.10)
  8. obs_action_logs_z_030   (labeled: turn left, vyaw=+0.30)
  9. obs_action_logs_z_n030  (labeled: turn right, vyaw=-0.30)

For each direction, you'll test env_0 through env_5 (6 environments each).

After each test, you'll record what the robot ACTUALLY did:
  - Forward, Backward, Strafe Left, Strafe Right, Turn Left, Turn Right, or Barely Moves

Press Enter after each test to continue.
""")

    input("Press Enter to begin comprehensive testing...")

    # Initialize controller
    controller = FixedMappingControllerV3("/home/ubuntu/mp2_mlp/policy_joyboy.pt")

    # Define test cases - ALL environments for ALL directions
    base_dir = "/home/ubuntu/debug"

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

    tests = []
    for dir_name, label in test_directions:
        for env_num in range(6):  # env_0 through env_5
            tests.append({
                "csv": f"{base_dir}/{dir_name}/env_{env_num}_actions.csv",
                "desc": f"{label} - env_{env_num}",
                "direction": dir_name,
                "env": env_num,
                "label": label
            })

    results = []
    total_tests = len(tests)

    # Run all tests
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/{total_tests}: {test['desc']}")
        print(f"LABELED AS: {test['label']}")
        print(f"{'='*70}")

        success = test_trajectory(controller, test['csv'], test['desc'])

        if success:
            print("\n[OBSERVATION] What did the robot ACTUALLY do?")
            print("Options:")
            print("  1. Forward")
            print("  2. Backward")
            print("  3. Strafe Left")
            print("  4. Strafe Right")
            print("  5. Turn Left (CCW)")
            print("  6. Turn Right (CW)")
            print("  7. Barely Moves / Unclear")
            print("  8. Combined Movement")

            observation = input("Enter number (1-8): ").strip()
            observation_map = {
                "1": "Forward",
                "2": "Backward",
                "3": "Strafe Left",
                "4": "Strafe Right",
                "5": "Turn Left (CCW)",
                "6": "Turn Right (CW)",
                "7": "Barely Moves",
                "8": "Combined Movement"
            }
            actual_behavior = observation_map.get(observation, "Unknown")
        else:
            actual_behavior = "FILE NOT FOUND"

        results.append({
            "test": test['desc'],
            "csv": test['csv'],
            "labeled": test['label'],
            "actual": actual_behavior,
            "success": success
        })

        if i < total_tests:
            input("\nPress Enter to continue to next test...")

    # Summary - Group by actual behavior
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST RESULTS")
    print("="*70)
    print(f"\nTotal Tests Run: {len(results)}\n")

    # Group results by actual behavior
    behavior_groups = {}
    for result in results:
        actual = result['actual']
        if actual not in behavior_groups:
            behavior_groups[actual] = []
        behavior_groups[actual].append(result)

    # Print grouped results
    for behavior in sorted(behavior_groups.keys()):
        tests = behavior_groups[behavior]
        print(f"\n{'='*70}")
        print(f"ACTUAL BEHAVIOR: {behavior} ({len(tests)} tests)")
        print(f"{'='*70}")

        for test in tests:
            match = "✓ MATCH" if behavior.upper() in test['labeled'].upper() else "✗ MISMATCH"
            print(f"  {match} | {test['test']:<40}")
            print(f"          Labeled: {test['labeled']}")
            print(f"          CSV: {test['csv']}")
            print()

    # Analysis section
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Find which environments consistently produce correct movements
    print("\n1. CORRECT MOVEMENTS (Label matches Actual):")
    correct_movements = [r for r in results if r['actual'].upper() in r['labeled'].upper()]
    if correct_movements:
        for result in correct_movements:
            print(f"   ✓ {result['test']} → {result['actual']}")
    else:
        print("   None found - significant transformation issue!")

    print("\n2. MISMATCHES (Label does NOT match Actual):")
    mismatches = [r for r in results if r['success'] and r['actual'].upper() not in r['labeled'].upper() and r['actual'] != "Barely Moves"]
    if mismatches:
        for result in mismatches:
            print(f"   ✗ {result['test']}")
            print(f"     Expected: {result['labeled']} | Got: {result['actual']}")
    else:
        print("   None - all movements matched expectations!")

    print("\n3. BARELY MOVES / UNCLEAR:")
    barely = [r for r in results if r['actual'] == "Barely Moves"]
    if barely:
        for result in barely:
            print(f"   ? {result['test']}")
    else:
        print("   None")

    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
Based on your results:

1. Use environments that MATCHED for training data
   - These have correct action → movement mappings

2. Investigate MISMATCHES to understand transformation errors:
   - If Forward → Backward: X-axis may be inverted
   - If Strafe Left → Strafe Right: Y-axis may be inverted
   - If Turn Left → Turn Right: Yaw-axis may be inverted
   - If Forward → Strafe: X and Y axes may be swapped

3. For environments that "Barely Move":
   - May have incorrect action scaling
   - May have joint limit clipping issues

4. Next step: Update policy training to use ONLY correct environments
""")

    print("\n[DONE] All tests complete - results saved above")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Test suite cancelled")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
