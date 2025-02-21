import streamlit as st
import cv2
import time
import numpy as np
import pyttsx3
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

# Function to get sign language video
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

# ✋ Sign to Speech
if role == "Sign to Speech":
    st.subheader("✋ Sign Language to Speech")
    cap = cv2.VideoCapture(0)

    if st.button("🎥 Start Camera"):
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        prediction_placeholder = st.empty()
        frames_list = []
        start_time = time.time()

        while True:
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

    if st.button("🛑 Stop Camera"):
        cap.release()

# 🎤 Speech to Sign
elif role == "Speech to Sign":
    st.subheader("🎤 Speech to Sign Language")

    # Using text input instead of microphone (compatible with Streamlit Cloud)
    text = st.text_input("📝 Enter text to convert to sign language:", "")

    if text:
        video_path = display_sign_video(text)
        if video_path:
            st.video(video_path)
        else:
            st.warning("⚠ No matching sign language video found.")
