#!/usr/bin/env python3

import rclpy
from geometry_msgs.msg import Twist
import time
import os

def main():
    # Initialize ROS2
    rclpy.init()
    
    # Create a simple node
    node = rclpy.create_node('pupper_controller')
    
    # Create publisher for cmd_vel
    cmd_vel_pub = node.create_publisher(Twist, '/cmd_vel', 10)
    
    # Make sure ROS2 environment is sourced
    if not os.environ.get('ROS_DISTRO'):
        print("Warning: ROS_DISTRO not set. Make sure to source your ROS2 setup!")
    
    print("Pupper controller started. Reading from velocity_command file...")
    print("Make sure you've run: ros2 launch mini_pupper_bringup bringup.launch.py")
    
    try:
        while rclpy.ok():
            ### TODO: Add your code here to receive the velocity command from the vision script and control the robot
            ### Read from the velocity_command file
            
            # Read yaw rate from the velocity_command file
            yaw_rate = 0.0
            try:
                with open("velocity_command", 'r') as file:
                    yaw_rate_str = file.readline().strip()
                    
                    if yaw_rate_str:
                        yaw_rate = float(yaw_rate_str)
                        print(f"Read yaw rate: {yaw_rate:.3f}")
                        
            except FileNotFoundError:
                # File doesn't exist yet, use 0.0
                pass
            except ValueError:
                # Invalid data in file, use 0.0
                yaw_rate = 0.0
            except Exception as e:
                print(f"Error reading velocity command: {e}")
                yaw_rate = 0.0
            
            # Limit yaw rate for safety
            max_yaw_rate = 1.0
            yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, yaw_rate))
            
            # Create and publish Twist message
            twist_msg = Twist()
            twist_msg.linear.x = 0.0
            twist_msg.linear.y = 0.0
            twist_msg.linear.z = 0.0
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            twist_msg.angular.z = yaw_rate
            
            cmd_vel_pub.publish(twist_msg)
            
            # Process ROS2 callbacks
            rclpy.spin_once(node, timeout_sec=0.01)
            
            # Sleep to match vision processing rate
            time.sleep(0.04)  # ~25 Hz
            
    except KeyboardInterrupt:
        print("\nStopping robot...")
        
    finally:
        # Send stop command
        stop_msg = Twist()
        cmd_vel_pub.publish(stop_msg)
        print("Sent stop command")
        
        # Cleanup
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()


# from pupper_controller.src.pupperv2 import pupper
# import math
# import time
# from absl import app

# def run_example():
#     pup = pupper.Pupper(run_on_robot=True,
#                         plane_tilt=0)
#     print("starting...")
#     pup.slow_stand(do_sleep=True)

#     yaw_rate = 0.0
#     try:
#         while True:
#             ### TODO: Add your code here to receive the velocity command from the vision script and control the robot
#             ### Read from the velocity_command file

#             pup.step(action={"x_velocity": 0.0,
#                                 "y_velocity": 0.0,
#                                 "yaw_rate": yaw_rate,
#                                 "height": -0.14,
#                                 "com_x_shift": 0.005})
#     finally:
#         pass

# def main(_):
#     run_example()

# app.run(main)   
