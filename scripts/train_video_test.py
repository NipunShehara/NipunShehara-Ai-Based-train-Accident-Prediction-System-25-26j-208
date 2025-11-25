from ultralytics import YOLO
import cv2
import os

MODEL_PATH = "runs/detect/train/weights/best.pt"
VIDEO_PATH = "test_media/videos/train.mp4"  # put a test video here

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        return

    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Test video not found: {VIDEO_PATH}")
        print("Put a file named train.mp4 inside test_media/videos/")
        return

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Train Detection (Press Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
