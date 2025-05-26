import cv2
import depthai as dai

pipeline = dai.Pipeline()

mono_cam = pipeline.createMonoCamera()
mono_cam.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
mono_cam.setFps(1)


xout_mono = pipeline.createXLinkOut()
xout_mono.setStreamName("mono")
mono_cam.out.link(xout_mono.input)


with dai.Device(pipeline) as device:
    mono_queue = device.getOutputQueue(name='mono', maxSize=4, blocking=False)

    while True:

        mono_frame = mono_queue.get().getCvFrame()

        cv2.imshow('mono', mono_frame)

        if cv2.waitKey(1) == ord('q'):
            break
