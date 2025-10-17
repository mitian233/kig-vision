# Quick smoke test for PiCamera OpenCV backend
from picamera_cv import PiCamera

cam = PiCamera()
print("Opened camera, resolution:", cam.resolution)
cam.start_preview()
cam.resolution = (320, 240)
print("Set resolution:", cam.resolution)
try:
    cam.capture("/tmp/test_capture.jpg")
    print("Captured /tmp/test_capture.jpg")
except Exception as e:
    print("Capture failed:", e)
finally:
    cam.close()
    print("Camera closed")
