# import time
# import numpy as np
# from MangDang.mini_pupper.ServoCalibration import MICROS_PER_RAD, NEUTRAL_ANGLE_DEGREES
# import MangDang.mini_pupper.nvram as nvram

# # Example function to set servo PWM values
# def set_servo_pwm(servo_id, angle_rad):
#     pulse_width = int(1500 + angle_rad * MICROS_PER_RAD)  # 1500 is the neutral pulse width
#     print(f"Setting servo {servo_id} to pulse width {pulse_width} for angle {angle_rad:.3f} radians")

# # Test the servos for all legs (assuming 12 servos total)
# def test_servo(servo_id, angle_rad):
#     print(f"Moving servo {servo_id} to {angle_rad:.3f} radians")
#     set_servo_pwm(servo_id, angle_rad)
#     time.sleep(1)  # Give time to observe the movement

# # Iterate through all servos
# def test_all_servos():
#     neutral_angle = 0  # Neutral position
#     small_angle = 0.1  # Small angle adjustment

#     # Example servo pins mapping (update based on your setup)
#     servo_pins = np.array([[15, 12, 9, 6], [14, 11, 8, 5], [13, 10, 7, 4]])

#     # Iterate through all servos, one by one
#     for row in range(3):
#         for col in range(4):
#             servo_id = servo_pins[row, col]
#             print(f"Testing servo on pin {servo_id} (row {row}, column {col})")
#             # Move the servo to a small angle and then back to neutral
#             test_servo(servo_id, small_angle)
#             test_servo(servo_id, neutral_angle)
#             time.sleep(1)  # Pause between tests for observation

#     # Once all servos have been tested, save the current neutral positions to NVRAM
#     nvram_data = {
#         'MICROS_PER_RAD': MICROS_PER_RAD,
#         'NEUTRAL_ANGLE_DEGREES': NEUTRAL_ANGLE_DEGREES
#     }

#     # Write the data to NVRAM
#     nvram.write(nvram_data)

#     print("Calibration data saved to NVRAM.")        

# # Run the test
# test_all_servos()
import time
import numpy as np
from MangDang.mini_pupper.ServoCalibration import MICROS_PER_RAD
import MangDang.mini_pupper.nvram as nvram

# Example function to set servo PWM values
def set_servo_pwm(servo_id, angle_rad):
    pulse_width = int(1500 + angle_rad * MICROS_PER_RAD)  # 1500 is the neutral pulse width
    print(f"Setting servo {servo_id} to pulse width {pulse_width} for angle {angle_rad:.3f} radians")

# Test the servos for all legs (assuming 12 servos total)
def test_servo(servo_id, angle_rad):
    print(f"Moving servo {servo_id} to {angle_rad:.3f} radians")
    set_servo_pwm(servo_id, angle_rad)
    time.sleep(1)  # Give time to observe the movement

# Iterate through all servos
def test_all_servos():
    neutral_angle = 0  # Neutral position
    small_angle = 0.1  # Small angle adjustment

    # Example servo pins mapping (update based on your setup)
    servo_pins = np.array([[15, 12, 9, 6], [14, 11, 8, 5], [13, 10, 7, 4]])

    # Initialize NEUTRAL_ANGLE_DEGREES to hold the correct angles
    NEUTRAL_ANGLE_DEGREES = np.zeros((3, 4))

    # Iterate through all servos, one by one
    for row in range(3):
        for col in range(4):
            servo_id = servo_pins[row, col]
            print(f"Testing servo on pin {servo_id} (row {row}, column {col})")
            # Move the servo to a small angle and then back to neutral
            test_servo(servo_id, small_angle)
            test_servo(servo_id, neutral_angle)
            
            # Dynamically set the NEUTRAL_ANGLE_DEGREES based on the current test
            NEUTRAL_ANGLE_DEGREES[row][col] = neutral_angle

            time.sleep(1)  # Pause between tests for observation

    # Once all servos have been tested, save the current neutral positions to NVRAM
    nvram_data = {
        'MICROS_PER_RAD': MICROS_PER_RAD,
        'NEUTRAL_ANGLE_DEGREES': NEUTRAL_ANGLE_DEGREES.tolist()  # Convert to list for JSON
    }

    # Write the data to NVRAM without calling .tolist() again
    nvram.write(nvram_data)

    print("Calibration data saved to NVRAM.")        

# Run the test
test_all_servos()





