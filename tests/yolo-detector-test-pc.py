import argparse
import logging
# from picamera2 import Picamera2
import time
import cv2
import numpy as np
import onnxruntime as ort
import boto3
import io
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Cat Detection Script")
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Enable debug mode for verbose output"
    )
    parser.add_argument(
        "-l", "--local-model",
        action="store_true",
        help="Enable local model"
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
    img = img.astype(np.float16)  # Convert to float16
    return img

def postprocess_detections(detections, frame_shape, conf_threshold=0.5):
    labels, boxes, scores = [], [], []
    h, w = frame_shape[:2]

    for det in detections:
        confidence = float(det[4])  # Convert confidence score to scalar
        class_probs = det[5:]  # Convert class ID to integer

        class_id = np.argmax(class_probs)  # Index of the class with the highest probability
        score = confidence # Confidence * max class probability
        logging.debug(f"class id: {class_id}")
        logging.debug(f"score: {score}")

        if score >= conf_threshold and class_id == 0:  # 15 = 'cat' in COCO
            x, y, bw, bh = det[:4]
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            labels.append("Cat")
            boxes.append((x1, y1, x2, y2))
            scores.append(score)

    return labels, boxes, scores

def use_aws_rekognition(frame):
    labels, boxes, scores = [], [], []
    h, w = frame.shape[:2]

    pil_image = Image.fromarray(frame)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")  # Ensure it's in JPEG format
    image_bytes = buffer.getvalue()

    client = boto3.client('rekognition')
    response = client.detect_labels(
        Image={'Bytes': image_bytes}, 
        MaxLabels=10,
        Features=['GENERAL_LABELS'])
    
    # Extract bounding boxes if available
    for label in response['Labels']:
        if label["Name"] != "Person":
            continue

        # Check if bounding box exists (some labels may not have one)
        if 'Instances' in label:
            for instance in label['Instances']:
                if 'BoundingBox' in instance:
                    bbox = instance['BoundingBox']
                    left = bbox["Left"]
                    top = bbox["Top"]
                    bw = bbox["Width"]
                    bh = bbox["Height"]
                    x1 = int(left * w)
                    y1 = int(top * h)
                    x2 = int((left + bw) * w)
                    y2 = int((top + bh) * h)
                    labels.append(label['Name'])
                    boxes.append((x1, y1, x2, y2))
                    scores.append(label['Confidence'])

        # cat_label = [label for label in response['Labels'] if label.get('Name') == 'People'] 

        # if (cat_label):
        #     if (cat_label[0]['Confidence']) > 75.0:
        #        logging.info("Found objects")

    return labels, boxes, scores



def draw_detections(frame, labels, boxes, scores):
    for label, (x1, y1, x2, y2), score in zip(labels, boxes, scores):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def capture_and_detect():
    args = parse_args()
    setup_logging(args.debug)

    # picam2 = Picamera2()
    # config = picam2.create_preview_configuration()
    # picam2.configure(config)
    # picam2.start()
    # time.sleep(2)  # Allow camera to adjust

    # Open the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()
    
    model_path = "yolov5m.onnx"  # Ensure the correct model path
    model = load_model(model_path)
    
    try:
        while True:
            # frame = picam2.capture_array()
            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Capture frame-by-frame
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if args.local_model:
                input_tensor = preprocess_frame(rgb_frame)
                
                detections = model.run(None, {model.get_inputs()[0].name: input_tensor})
                # if args.debug:
                #     logging.debug(f"detections.shape: {len(detections)}")  # Show dimensions
                #     logging.debug(f"detections[0].shape: {detections[0].shape}")  # Show dimensions
                #     logging.debug(f"detections[1].shape: {detections[1].shape}")  # Show dimensions
                #     logging.debug(f"detections[2].shape: {detections[2].shape}")  # Show dimensions
                #     logging.debug(f"detections[3].shape: {detections[3].shape}")  # Show dimensions
                labels, boxes, scores = postprocess_detections(detections[0][0], rgb_frame.shape)
                draw_detections(rgb_frame, labels, boxes, scores)
            else:
                labels, boxes, scores = use_aws_rekognition(rgb_frame)
                draw_detections(rgb_frame, labels, boxes, scores)
            
            cv2.imshow('Cat Detector', rgb_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # picam2.stop()
        cap.release()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_detect()
