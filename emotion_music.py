import cv2
import numpy as np
import time
import subprocess
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model("fast_model.h5")

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Haarcascade for face detection
face_cascade = cv2.CascadeClassifier("/home/pi/.local/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml")

# YouTube video links for different emotions
emotion_videos = {
    "Happy": "https://youtu.be/aPEdQ0G8GtY?si=8q_V40naK5IrH4jy",
    "Sad": "https://youtu.be/pRpeEdMmmQ0?si=J42M4Dzx5-zhRUco",
    "Neutral": "https://youtu.be/vb3rQ9VDRKc?si=WZTDJML-4WECEcfr",
    "Surprise": "https://www.youtube.com/watch?v=4NRXx6U8ABQ",
    "Disgust": "https://youtu.be/oSpMspvMkSQ?si=5mJGDg2Z4TM3O_Sn",
    "Fear": "https://www.youtube.com/watch?v=GJY8jJkDoMY",
    "Angry": "https://youtu.be/5WDQgBGfx40?si=21wcCfyoc0HkbC2I"
}

# Google Home device name
GOOGLE_HOME_DEVICE = "Hall speaker"

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break  

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    detected_emotion = None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))  
        roi_gray = np.stack((roi_gray,)*3, axis=-1)  
        roi_gray = roi_gray / 255.0  
        roi_gray = np.expand_dims(roi_gray, axis=0)  

        prediction = model.predict(roi_gray)
        detected_emotion = emotion_labels[np.argmax(prediction)]
        print(f"Detected Emotion: {detected_emotion}")  

        # Draw emotion label on the screen
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, detected_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

    # If emotion detected, play the corresponding video
    if detected_emotion and detected_emotion in emotion_videos:
        video_url = emotion_videos[detected_emotion]
        print(f"Playing {detected_emotion} video: {video_url}")
        
        # Command to cast YouTube video to Google Home
        command = f'catt -d "{GOOGLE_HOME_DEVICE}" cast "{video_url}"'
        subprocess.run(command, shell=True)
        
        # Wait for 5 minutes before detecting again
        print("Waiting for 5 minutes before detecting again...")
        time.sleep(300)  # 300 seconds = 5 minutes

cap.release()
cv2.destroyAllWindows()