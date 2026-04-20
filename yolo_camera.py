# Import the YOLO model from ultralytics library
from ultralytics import YOLO

# Import opencv library - used to access camera and show video
import cv2

# Load the YOLOv8 nano model (smallest and fastest model)
# It will auto-download on first run
model = YOLO("yolov8n.pt")
# print(model)
# Open your laptop camera (0 means default camera)
cap = cv2.VideoCapture(0)

# Set camera capture resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)   # width in pixels
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)   # height in pixels

# Start an infinite loop to keep reading camera frames
while True:

     # Read one frame from the camera
     # ret = True if frame was read successfully
     # frame = the actual image/frame
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (1280, 720))
    

     # If camera frame was not read, skip and try again
    if not ret:
        break

    # Run YOLO detection on the current frame
     # stream=True makes it faster and memory efficient
    results = model(frame, stream=True)

     # Loop through each detection result
    for result in results:

         # Draw the detection boxes on the frame automatically
        frame = result.plot()

     # Show the frame in a window called "YOLO Camera"
    cv2.imshow("YOLO Camera", frame)

     # Wait for 1ms and check if user pressed 'q' to quit
    # ord('q') gets the keyboard code for q
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera so other apps can use it
cap.release()

# Close all opencv windows
cv2.destroyAllWindows()