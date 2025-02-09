from picamera2 import Picamera2, Preview
import time
import cv2
import numpy as np

def capture_and_stream():
    # Initialize the camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    
    # Start the camera
    picam2.start()
    time.sleep(2)  # Allow camera to adjust
    
    frame_count = 0
    
    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save each frame
            filename = f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(filename, frame)
            frame_count += 1
            
            # Display the live stream
            cv2.imshow('Camera Feed', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_stream()
