import argparse
import logging
from picamera2 import Picamera2
import time
import cv2
import numpy as np
import onnxruntime as ort

def parse_args():
    parser = argparse.ArgumentParser(description="Cat Detection Script")
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode for verbose output"
    )
    return parser.parse_args()

def setup_logging(debug):
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def load_model(onnx_model_path):
    return ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

def preprocess_frame(frame, input_size=(640, 640)):
    img = cv2.resize(frame, input_size)
    img = img.astype(np.float32) / 255.0  # Normalize
    img = np.transpose(img, (2, 0, 1))  # Change channel order
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def postprocess_detections(detections, frame_shape, conf_threshold=0.5):
    boxes, scores = [], []
    h, w = frame_shape[:2]

    for det in detections[0]:
        confidence = det[4].item()  # Ensure it's a scalar
        class_id = int(det[5].item())  # Ensure it's an integer
        
        if confidence >= conf_threshold and class_id == 15:  # 15 = 'cat' in COCO
            x, y, bw, bh = det[:4]
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            boxes.append((x1, y1, x2, y2))
            scores.append(confidence)

    return boxes, scores


def draw_detections(frame, boxes, scores):
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Cat: {score:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def capture_and_detect():
    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Allow camera to adjust
    
    model_path = "yolov5n.onnx"  # Ensure the correct model path
    model = load_model(model_path)
    
    try:
        while True:
            frame = picam2.capture_array()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess_frame(rgb_frame)
            
            detections = model.run(None, {model.get_inputs()[0].name: input_tensor})
            boxes, scores = postprocess_detections(detections, frame.shape)
            draw_detections(frame, boxes, scores)
            
            cv2.imshow('Cat Detector', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_detect()
