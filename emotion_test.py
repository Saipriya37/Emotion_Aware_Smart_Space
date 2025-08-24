import cv2

# Correct path to Haarcascade file
cascade_path = "/home/pi/.local/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml"

# Load Haarcascade
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print("Error: Haarcascade file not found!")
else:
    print("Haarcascade loaded successfully!")
import cv2, numpy as np
from tensorflow.keras.models import load_model

model = load_model("fast_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
face_cascade = cv2.CascadeClassifier("/home/pi/.local/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: print("Failed to capture image"); break  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))  # Resize to 64x64
        roi_gray = np.stack((roi_gray,)*3, axis=-1)  # Convert grayscale to 3 channels
        roi_gray = roi_gray / 255.0  # Normalize
        roi_gray = np.expand_dims(roi_gray, axis=0)  # Expand dimensions for model

        prediction = model.predict(roi_gray)
        print(f"Raw Prediction: {prediction}")  
        emotion = emotion_labels[np.argmax(prediction)]
        print(f"Detected Emotion: {emotion}")  
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break  

cap.release()
cv2.destroyAllWindows()