import cv2
import numpy as np
import onnxruntime as ort
import argparse
import logging

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

def main():
    args = parse_args()
    setup_logging(args.debug)

    logging.info("Starting the cat detection process.")

    # Load the YOLOv5 model
    model_path = "yolov5n.onnx"
    session = ort.InferenceSession(model_path)

    # Define class names (COCO dataset, class 15 is 'cat')
    CLASSES = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat"]

    # Set up camera
    cap = cv2.VideoCapture(0)
    if args.debug:
        # Check if the capture device was successfully opened
        if not cap.isOpened():
            logging.debug("Error: Could not open video capture device.")
        else:
            logging.debug("Video capture device opened successfully.")

    while True:
        if args.debug:
            logging.debug("keep capturing...")
        ret, frame = cap.read()
        if not ret:
            if args.debug:
                logging.debug("cap.read unsuccess")
            break

        # Preprocess the image
        img = cv2.resize(frame, (640, 640))
        img = img.transpose(2, 0, 1)  # Convert HWC to CHW
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  # Normalize

        if args.debug:
            logging.debug(f"First pixel value (BGR): {img[0, 0]}")

        # Run inference
        inputs = {session.get_inputs()[0].name: img}
        outputs = session.run(None, inputs)

        # Process detections
        detections = outputs[0][0]  # YOLO outputs bounding boxes, class IDs, confidence

        for detection in detections:
            x, y, w, h, conf, class_id = detection[:6]
            if conf > 0.5 and int(class_id) == 15:  # Class 15 = cat
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Cat Detected!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show video feed
        cv2.imshow("Cat Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Cat detection process completed.")

if __name__ == "__main__":
    main()