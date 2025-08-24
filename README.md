# Emotion_Aware_Smart_Space

🌟 Emotion-Aware Smart Space

An AI + IoT project that detects human emotions in real-time and adapts the environment by playing mood-based music, adjusting lighting, and creating a personalized smart space experience.

📖 Table of Contents

Inspiration
Features
Tech Stack
System Architecture
Demo
Challenges
Real-world example
Future Improvements
Conclusion 

💡 Inspiration

Smart homes today respond to commands, but they don’t really understand the user. After a long stressful day, you still need to say “play music” or “turn off the lights.” We wanted to build a smart space that understands your emotions and adapts automatically to enhance comfort and well-being.

✨ Features

Emotion Detection 🧠: Detects emotions from facial expressions in real-time using Raspberry Pi + camera.

Mood-Based Music 🎶: Plays songs that match your emotional state via Google Home.

Smart Adjustments 🌱: Controls lighting, air quality, and ambiance based on detected mood.

Seamless Integration 💡: Works with IoT devices and Google Home for a fully immersive experience.

🛠 Tech Stack

Languages: Python

AI/ML: TensorFlow / Keras, OpenCV

Hardware: Raspberry Pi, Camera Module, IoT Sensors (light, motion, air quality)

Smart Integration: Google Home API / catt

Other Tools: Streamlit (for testing UI)

🔗 System Architecture

[Camera + Sensors] ---> [Emotion Detection Model] ---> [Smart Controller (Raspberry Pi)] 
         |                           |                          |
         |                           v                          v
         |                [Emotion Classification]    [IoT Devices + Google Home]
         |                           |                          |
         ------------------> [Personalized Smart Space] <--------


🎥 Demo
https://drive.google.com/file/d/1xdKGyUMjw9dUJ_HooxPy0tkB1BnERPMY/view?usp=sharing

🚧 Challenges

Running real-time emotion detection on a lightweight Raspberry Pi.

Integrating IoT devices with AI outputs.

Synchronizing Google Home automation with detected emotions.

🚀 Real-World Example: How Emotion-Aware Smart Space Works

Imagine you come home after a long day at college or work. You’re feeling tired and a bit low. As soon as you enter your room:

📷 The camera detects your emotion (sad/tired) using AI.

💡 Instantly, the IoT lights adjust to a soft, warm glow.

🎶 Your Google Home plays relaxing music or a motivational YouTube video.

🌬️ The fan or AC adapts to a comfortable level.

Now, if you walk in another day feeling happy or excited:

Lights become brighter,

Music turns cheerful,

The environment adapts to your mood automatically.

This makes your room not just “smart” but emotionally intelligent — turning tech into a personal companion.

🎉 Future Improvements

Multi-room support for a complete smart home experience.

Adding biometric sensors (heart rate, skin conductance) for deeper emotional analysis.

Voice-enabled emotional assistant for relaxation or productivity suggestions.

Scalable cloud integration for faster processing.

🎯 Conclusion

The Emotion-Aware Smart Space project demonstrates how Artificial Intelligence and IoT can be combined to create environments that adapt intelligently to human emotions. By detecting real-time moods and personalizing surroundings such as lighting, music, and ambiance, this project shows the potential of building empathetic, human-centered smart spaces.

This is not just a tech demo — it’s a step toward the future of living spaces where technology doesn’t just respond to commands, but truly understands and cares for the user’s well-being.


