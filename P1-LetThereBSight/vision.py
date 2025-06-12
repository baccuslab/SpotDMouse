#!/usr/bin/env python3

from pathlib import Path
# import cv2
import depthai as dai
import numpy as np
import time
import argparse

def send_velocity_command(yaw_velocity):
    ### TODO: Add your code here to send the velocity command to the pupper
    ### Write the velocity command to the file "velocity_command"
    try:
        with open("velocity_command", 'w') as file:
            file.write(str(yaw_velocity))
        print(f"Yaw velocity {yaw_velocity:.3f} written to velocity_command file")
    except Exception as e:
        print(f"Error writing velocity command to file: {e}")

nnPathDefault = str((Path(__file__).parent / Path('mobilenet-ssd_openvino_2021.4_6shave.blob')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('nnPath', nargs='?', help="Path to mobilenet detection network blob", default=nnPathDefault)
parser.add_argument('-s', '--sync', action="store_true", help="Sync RGB output with NN output", default=False)
args = parser.parse_args()

if not Path(nnPathDefault).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# MobilenetSSD label texts
labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)
nnNetworkOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")
nnNetworkOut.setStreamName("nnNetwork");

# Properties
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setFps(20)
# Define a neural network that will make predictions based on the source frames
nn.setConfidenceThreshold(0.1)
nn.setBlobPath(args.nnPath)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Linking
if args.sync:
    nn.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

camRgb.preview.link(nn.input)
nn.out.link(nnOut.input)
nn.outNetwork.link(nnNetworkOut.input);

# Connect to device and start pipeline
try:
    with dai.Device(pipeline) as device:
        # Output queues will be used to get the rgb frames and nn data from the outputs defined above
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        qNN = device.getOutputQueue(name="nnNetwork", maxSize=4, blocking=False);

        frame = None
        detections = []
        startTime = time.monotonic()
        counter = 0
        color2 = (255, 255, 255)
        printOutputLayersOnce = True
        yaw_velocity = 0.0
        
        # Proportional controller gain
        KP = 1.0
        last_frame_counter = -1
        
        while True:
            if args.sync:
                inRgb = qRgb.get()
                inDet = qDet.get()
                inNN = qNN.get()
            else:
                inRgb = qRgb.tryGet()
                inDet = qDet.tryGet()
                inNN = qNN.tryGet()

            if inDet is not None:
                detections = inDet.detections
                counter += 1
                
                # Skip if we've already processed this frame
                if counter == last_frame_counter:
                    continue
                last_frame_counter = counter
                
                ### TODO: Add your code here to detect a person and run a visual servoing controller
                ### Steps:
                ### 1. Detect a person by pulling out the labelMap bounding box with the correct label text
                ### 2. Compute the x midpoint of the bounding box
                ### 3. Compute the error between the x midpoint and the center of the image (the bounds of the image are normalized to be 0 to 1).
                ### 4. Compute the yaw rate command using a proportional controller
                ### 5. Send the yaw rate command to the pupper
                
                # Step 2: Print detections to see what we're getting
                print(f"Frame {counter}: {len(detections)} total detections")
                
                # Step 1: Filter detections for "person" class
                person_detections = []
                for detection in detections:
                    label_text = labelMap[detection.label]
                    print(f"  Detection: {label_text} (confidence: {detection.confidence:.2f})")
                    
                    if label_text == "person":
                        person_detections.append(detection)
                
                print(f"Found {len(person_detections)} person detection(s)")
                
                if len(person_detections) > 0:
                    # If multiple people detected, choose the one with largest bounding box (closest)
                    largest_person = max(person_detections, 
                                    key=lambda d: (d.xmax - d.xmin) * (d.ymax - d.ymin))
                    
                    # Step 2: Compute the x midpoint of the bounding box
                    x_midpoint = (largest_person.xmin + largest_person.xmax) / 2.0
                    print(f"Person x_midpoint: {x_midpoint:.3f}")
                    
                    # Step 3: Compute error between x midpoint and center of image
                    # Center of normalized image is 0.5
                    image_center = 0.5
                    x_error = x_midpoint - image_center
                    print(f"X error (person - center): {x_error:.3f}")
                    
                    # Step 4: Compute yaw rate using proportional controller
                    yaw_velocity = KP * x_error
                    print(f"Yaw velocity command: {yaw_velocity:.3f}")
                    
                    # Optional: Limit yaw velocity to reasonable range
                    max_yaw_rate = 2.0  # Adjust as needed
                    yaw_velocity = max(-max_yaw_rate, min(max_yaw_rate, yaw_velocity))
                    
                else:
                    # No person detected, stop turning
                    yaw_velocity = 0.0
                    print("No person detected - stopping rotation")
                
                # Step 5: Send the yaw rate command to the pupper
                send_velocity_command(yaw_velocity)

                frame = inRgb.getCvFrame()
                # Draw the bounding box
                x1 = int(largest_person.xmin * frame.shape[1])
                y1 = int(largest_person.ymin * frame.shape[0])
                x2 = int(largest_person.xmax * frame.shape[1])
                y2 = int(largest_person.ymax * frame.shape[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"person: {largest_person.confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save the frame
                timestamp = int(time.time())
                filename = f"snapshot_person_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved snapshot to {filename}")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)

except KeyboardInterrupt:
    print("Interrupted by user, shutting down...")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    print("Cleanup completed")