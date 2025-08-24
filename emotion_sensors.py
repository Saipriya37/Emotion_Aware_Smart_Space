import cv2
import numpy as np
import spidev
import time
import RPi.GPIO as GPIO
from tensorflow.keras.models import load_model

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Define GPIO Pins
RGB_PINS = {'Red': 17, 'Green': 27, 'Blue': 22}
FAN_PIN = 23
TEMP_SENSOR_CH = 2
SOUND_SENSOR_CH = 0
LIGHT_SENSOR_CH = 1
HEART_RATE_SENSOR_CH = 3
AIR_QUALITY_SENSOR_CH = 4
MOTION_SENSOR_PIN = 24  # PIR motion sensor

# Initialize GPIO Pins
for pin in RGB_PINS.values():
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

GPIO.setup(FAN_PIN, GPIO.OUT)
GPIO.setup(MOTION_SENSOR_PIN, GPIO.IN)  # PIR Motion Sensor
fan_pwm = GPIO.PWM(FAN_PIN, 100)
fan_pwm.start(0)

# MCP3008 SPI Setup
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

# Function to read ADC
def read_adc(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    return ((adc[1] & 3) << 8) + adc[2]

# Function to categorize sensor values
def categorize_sound(value):
    if value < 100:
        return "Low"
    elif 100 <= value < 500:
        return "Normal"
    else:
        return "High"

def categorize_light(value):
    if value < 150:
        return "Dark"
    elif 150 <= value < 600:
        return "Normal"
    else:
        return "Bright"

def categorize_heart_rate(value):
    if value < 300:
        return "Low"
    elif 300 <= value < 700:
        return "Normal"
    else:
        return "High"

def categorize_air_quality(value):
    if value < 200:
        return "Good"
    elif 200 <= value < 500:
        return "Moderate"
    else:
        return "Poor"

# Load Haarcascade for Face Detection
cascade_path = "/home/pi/.local/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# Load Emotion Model
model = load_model("fast_model.h5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Tracking user presence
user_present = False
emotion_recorded = False  # Ensure one reading per user entry

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

    motion_detected = GPIO.input(MOTION_SENSOR_PIN)  # Read PIR Motion Sensor

    if len(faces) > 0 and not emotion_recorded:
        user_present = True
        x, y, w, h = faces[0]  # Take only the first detected face
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (64, 64))
        roi_gray = np.stack((roi_gray,) * 3, axis=-1) / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)

        prediction = model.predict(roi_gray)
        emotion = emotion_labels[np.argmax(prediction)]
        print(f"Detected Emotion: {emotion}")

        # Control Sensors Based on Emotion
        GPIO.output(RGB_PINS['Red'], GPIO.LOW)
        GPIO.output(RGB_PINS['Green'], GPIO.LOW)
        GPIO.output(RGB_PINS['Blue'], GPIO.LOW)

        if emotion == "Happy":
            GPIO.output(RGB_PINS['Green'], GPIO.HIGH)
            fan_pwm.ChangeDutyCycle(100)  # Full speed
        elif emotion == "Sad":
            GPIO.output(RGB_PINS['Blue'], GPIO.HIGH)
            fan_pwm.ChangeDutyCycle(50)
        elif emotion == "Angry":
            GPIO.output(RGB_PINS['Red'], GPIO.HIGH)
            fan_pwm.ChangeDutyCycle(70)
        else:
            fan_pwm.ChangeDutyCycle(30)

        # Read Sensor Data Once
        sound_level = read_adc(SOUND_SENSOR_CH)
        light_level = read_adc(LIGHT_SENSOR_CH)
        temp_value = read_adc(TEMP_SENSOR_CH)
        temperature = (temp_value * 3.3 / 1023) * 100
        heart_rate = read_adc(HEART_RATE_SENSOR_CH)
        air_quality = read_adc(AIR_QUALITY_SENSOR_CH)

        # Categorize values
        sound_status = categorize_sound(sound_level)
        light_status = categorize_light(light_level)
        heart_rate_status = categorize_heart_rate(heart_rate)
        air_quality_status = categorize_air_quality(air_quality)

        print(f"Sound Level: {sound_status}, Light Level: {light_status}, Temperature: {temperature:.2f}°C")
        print(f"Heart Rate: {heart_rate_status}, Air Quality: {air_quality_status}")
        print(f"Motion Detected: {'Yes' if motion_detected else 'No'}")

        # Mark that emotion has been recorded once
        emotion_recorded = True

        # Draw Face Rectangle and Label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    elif len(faces) == 0 and user_present:
        print("User has left, resetting for next detection...")
        user_present = False
        emotion_recorded = False  # Reset for next entry
        for pin in RGB_PINS.values():
            GPIO.output(pin, GPIO.LOW)
        fan_pwm.ChangeDutyCycle(0)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
fan_pwm.stop()