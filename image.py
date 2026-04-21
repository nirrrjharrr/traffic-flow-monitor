import cv2

cap = cv2.VideoCapture("videos/traffic_main.mp4")

ret, frame = cap.read()

if ret:
    cv2.imwrite("frame.jpg", frame)
    print("Frame saved as frame.jpg")

cap.release()