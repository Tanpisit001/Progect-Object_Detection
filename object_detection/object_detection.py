import os
import csv
import time
from collections import Counter
from datetime import datetime
import cv2
import torch
from ultralytics import YOLO

# Camera FPS is a request, not a guaranteed AI processing speed.
OUTPUT = "output"
CAMERA_FPS = 60
CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
IMAGE_SIZE = 320
VIDEO_FPS = 15
os.makedirs(OUTPUT, exist_ok=True)

# Use CUDA GPU when PyTorch supports it; otherwise use CPU.
device = 0 if torch.cuda.is_available() else "cpu"
model = YOLO("yolo11n.pt")
detect_every = 1 if device == 0 else 2
print("AI device:", "CUDA GPU" if device == 0 else "CPU")

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    raise RuntimeError("Cannot open webcam")
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
camera.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
print("Camera reported FPS:", camera.get(cv2.CAP_PROP_FPS))

confidence = 0.50
detection_on = True
recording = False
writer = None
record_start = 0.0
frames_written = 0
frame_number = 0
last_boxes = []
last_time = time.perf_counter()
fps = 0.0
log_path = os.path.join(OUTPUT, "detections.csv")
if not os.path.exists(log_path):
    with open(log_path, "w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerow(["Time", "Object", "Count"])


def show_text(image, message, x, y, color=(0, 255, 0)):
    cv2.putText(image, message, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Cannot read webcam frame")
            break
        frame_number += 1
        now = datetime.now()

        # Skip some AI calls on CPU, but still display fresh camera frames.
        if detection_on and frame_number % detect_every == 0:
            result = model.predict(frame, imgsz=IMAGE_SIZE,
                                   conf=confidence, device=device,
                                   verbose=False)[0]
            last_boxes = []
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                name = model.names[int(box.cls[0])]
                score = float(box.conf[0])
                last_boxes.append((x1, y1, x2, y2, name, score))
        if not detection_on:
            last_boxes = []

        counts = Counter(item[4] for item in last_boxes)
        for x1, y1, x2, y2, name, score in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_y = max(20, y1 - 8)
            show_text(frame, f"{name} {score:.2f}", x1, label_y)

        current = time.perf_counter()
        instantaneous_fps = 1 / max(current - last_time, 0.000001)
        fps = instantaneous_fps if fps == 0 else fps * 0.8 + instantaneous_fps * 0.2
        last_time = current
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (330, 125), (35, 35, 35), -1)
        show_text(frame, f"Objects: {sum(counts.values())}", 10, 25)
        show_text(frame, f"Display FPS: {fps:.1f}", 10, 49)
        show_text(frame, f"Confidence: {confidence:.2f}", 10, 73)
        show_text(frame, f"Detection: {'ON' if detection_on else 'OFF'}", 10, 97)
        show_text(frame, now.strftime("%H:%M:%S"), 10, 120)
        for index, (name, count) in enumerate(counts.most_common(4)):
            show_text(frame, f"{name}: {count}", width - 150, 25 + index * 24)
        show_text(frame, "Q Quit | S Photo | R Record | D Detect | L Log | +/- Conf",
                  8, height - 12, (255, 255, 255))

        if recording:
            show_text(frame, "REC", width - 70, height - 35, (0, 0, 255))
            target = int((current - record_start) * VIDEO_FPS)
            while frames_written < target:
                writer.write(frame)
                frames_written += 1

        cv2.imshow("YOLO11 Object Detection - Fast", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            filename = os.path.join(OUTPUT, now.strftime("image_%Y%m%d_%H%M%S_%f.jpg"))
            print("Saved image" if cv2.imwrite(filename, frame) else "Image save failed")
        elif key == ord("d"):
            detection_on = not detection_on
        elif key == ord("r"):
            if recording:
                recording = False
                writer.release()
                writer = None
                print("Video saved")
            else:
                filename = os.path.join(OUTPUT, now.strftime("video_%Y%m%d_%H%M%S.mp4"))
                writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"),
                                         VIDEO_FPS, (width, height))
                recording = writer.isOpened()
                record_start = time.perf_counter()
                frames_written = 0
                if not recording:
                    writer.release()
                    writer = None
                print("Recording started" if recording else "Video writer failed")
        elif key == ord("l"):
            with open(log_path, "a", newline="", encoding="utf-8") as file:
                for name, count in (counts or {"none": 0}).items():
                    csv.writer(file).writerow([now.isoformat(timespec="seconds"), name, count])
            print("Log saved")
        elif key in (ord("+"), ord("=")):
            confidence = min(0.95, round(confidence + 0.05, 2))
        elif key == ord("-"):
            confidence = max(0.05, round(confidence - 0.05, 2))
finally:
    camera.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
