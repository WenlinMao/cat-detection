from picamera2 import Picamera2
from picamera2 import Preview
import time

# Initialize the camera
picam2 = Picamera2()

# Configure the camera (you can adjust these settings based on your needs)
picam2.start_preview(Preview.NULL)

# Allow the camera to adjust to the lighting for a short period
time.sleep(2)

# Capture the image and save it as 'image.jpg'
picam2.capture_file("image.jpg")

# Optionally, stop the preview
picam2.stop_preview()

print("Image captured and saved as 'image.jpg'.")