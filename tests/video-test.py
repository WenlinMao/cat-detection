import time
import picamera

# Initialize the camera
with picamera.PICamera() as camera:
    # Set the resolution of the camera
    camera.resolution = (1024, 768)
    
    # Wait for the camera to warm up
    print("Warm up the camera...")
    time.sleep(2)
    
    # Capture an image and save it to a file
    camera.capture('image.jpg')
    print("Image captured and saved as 'image.jpg'")