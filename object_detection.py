import cv2
import torch
import pyttsx3

# YOLOv5 model
print("Loading YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
print("Model loaded successfully.")

# text-to-speech engine
engine = pyttsx3.init()
print("Text-to-speech engine initialized.")

# camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Camera opened successfully.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # inference on the frame
        results = model(frame)

        # render results on the frame
        results.render()

        # conversion of results to string for text-to-speech
        detected_objects = results.names
        for obj in results.xyxy[0]:  # Get detections
            class_id = int(obj[5])  # Object class ID
            object_name = detected_objects[class_id]
            print(f"Detected: {object_name}")

            # speak the detected object
            engine.say(object_name)
            engine.runAndWait()

        # display the frame with detections
        cv2.imshow("Camera", frame)

        # exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
