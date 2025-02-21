import streamlit as st
import cv2
import time
import numpy as np
import pyttsx3
import speech_recognition as sr
import sounddevice as sd
from tensorflow.keras.models import load_model
import os

# Constants
SEQUENCE_LENGTH = 30  # Number of frames per sequence
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64  # Resizing dimensions
CLASSES_LIST = ["Namaste", "Good Afternoon", "Good Night", "How are you"]
DATA_FOLDER = "data"  # Folder containing sign videos

# Load model
LRCN_model = load_model("LRCN_model.h5")

# Function to predict sign action
def predict_single_action(frames_list):
    frames_array = np.expand_dims(frames_list, axis=0)
    predicted_probs = LRCN_model.predict(frames_array)[0]
    predicted_label = np.argmax(predicted_probs)
    return CLASSES_LIST[predicted_label], predicted_probs[predicted_label]

# Text-to-speech function
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Speech-to-text function using sounddevice
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening...")
        try:
            audio = recognizer.listen(source)
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            st.warning("❌ Could not understand the audio.")
        except sr.RequestError:
            st.error("⚠ Speech service unavailable.")
    return None

# Display sign video
def display_sign_video(sign_text):
    video_path = os.path.join(DATA_FOLDER, f"{sign_text}.mp4")
    return video_path if os.path.exists(video_path) else None

# Streamlit UI
st.set_page_config(page_title="Sign-Speech Communication", layout="wide")
st.title("🧏‍♂️ Sign-Speech Communication")
st.markdown("A system enabling seamless interaction between special and normal persons.")

# Role Selection
role = st.radio("Select Mode:", ["Sign to Speech", "Speech to Sign"], horizontal=True)
st.divider()

# Webcam Setup with Session State
if "camera_active" not in st.session_state:
    st.session_state.camera_active = False

if role == "Sign to Speech":
    st.subheader("✋ Sign Language to Speech")

    # Start Camera Button
    if st.button("🎥 Start Camera"):
        st.session_state.camera_active = True

    # Stop Camera Button
    if st.button("🛑 Stop Camera"):
        st.session_state.camera_active = False

    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        prediction_placeholder = st.empty()
        frames_list = []
        start_time = time.time()

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("⚠ Camera error.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame, channels="RGB")

            elapsed_time = time.time() - start_time
            progress_bar.progress(min(int((elapsed_time / 4) * 100), 100))

            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH)) / 255
            frames_list.append(resized_frame)

            if elapsed_time >= 4:
                if len(frames_list) >= SEQUENCE_LENGTH:
                    predicted_class, confidence = predict_single_action(frames_list[-SEQUENCE_LENGTH:])
                    prediction_placeholder.success(f"✅ Predicted: {predicted_class} ({confidence:.2f})")
                    speak(predicted_class)
                start_time = time.time()
                frames_list = []

        cap.release()

elif role == "Speech to Sign":
    st.subheader("🎤 Speech to Sign Language")

    if st.button("🎙 Start Speaking"):
        text = recognize_speech()
        if text:
            video_path = display_sign_video(text)
            if video_path:
                st.video(video_path)
            else:
                st.warning("⚠ No matching sign language video found.")
