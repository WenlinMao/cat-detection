import argparse
import logging
from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort
import boto3
import io
from PIL import Image
import pygame
import time

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

def postprocess_detections(detections, frame_shape, sound, conf_threshold=0.5):
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
            play_audio(sound)

    return labels, boxes, scores

def use_aws_rekognition(frame, sound):
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
        if label["Name"] != "Cat" or label['Confidence'] <= 75.0:
            continue
        
        play_audio(sound)
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

    return labels, boxes, scores

def draw_detections(frame, labels, boxes, scores):
    for label, (x1, y1, x2, y2), score in zip(labels, boxes, scores):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}: {score:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def play_audio(sound):
    # Play the sound for 1000 milliseconds (1 second)
    sound.play(maxtime=1000)
    
    # Keep the program running long enough to hear the sound
    pygame.time.delay(1000)

def capture_and_detect():
    args = parse_args()
    setup_logging(args.debug)

    # Initialize the mixer module
    pygame.mixer.init()
    # Load the sound file
    sound = pygame.mixer.Sound('dog.wav')

    picam2 = Picamera2()
    config = picam2.create_preview_configuration()
    picam2.configure(config)
    picam2.start()
    time.sleep(2)  # Allow camera to adjust
    
    model_path = "yolov5n.onnx"  # Ensure the correct model path
    model = load_model(model_path)

    frame_count = 0
    frame_interval = 10
    
    try:
        while True:
            frame = picam2.capture_array()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if args.local_model:
                logging.info("Using Local Model")
                input_tensor = preprocess_frame(rgb_frame)
                
                detections = model.run(None, {model.get_inputs()[0].name: input_tensor})
                # if args.debug:
                #     logging.debug(f"detections.shape: {len(detections)}")  # Show dimensions
                #     logging.debug(f"detections[0].shape: {detections[0].shape}")  # Show dimensions
                #     logging.debug(f"detections[1].shape: {detections[1].shape}")  # Show dimensions
                #     logging.debug(f"detections[2].shape: {detections[2].shape}")  # Show dimensions
                #     logging.debug(f"detections[3].shape: {detections[3].shape}")  # Show dimensions
                labels, boxes, scores = postprocess_detections(detections[0][0], rgb_frame.shape, sound)
            else:
                logging.info("Using AWS Model")
                labels, boxes, scores = use_aws_rekognition(rgb_frame, sound)
            
            draw_detections(rgb_frame, labels, boxes, scores)

            # cv2.imshow('Cat Detector', rgb_frame)
            if frame_count % frame_interval == 0:
                filename = f"frame_{image_count:04d}.jpg"
                cv2.imwrite(filename, frame)
                image_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_detect()
