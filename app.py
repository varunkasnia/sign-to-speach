import streamlit as st
import av
import numpy as np
import pyttsx3
import speech_recognition as sr
import os
import time
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Constants
SEQUENCE_LENGTH = 30  # Number of frames per sequence
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64  # Resizing dimensions
CLASSES_LIST = ["Namaste", "Good Afternoon", "Good Night", "How are you"]  # Replace with your class names
DATA_FOLDER = "data"  # Folder containing sign videos

# Load your pre-trained LRCN model
LRCN_model = load_model("LRCN_model.h5")  # Replace with your model path

# Function to predict action
def predict_single_action(frames_list):
    frames_array = np.expand_dims(frames_list, axis=0)
    predicted_labels_probabilities = LRCN_model.predict(frames_array)[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    return predicted_class_name, predicted_labels_probabilities[predicted_label]

# Text-to-speech function
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Speech-to-text function
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
            return None
        except sr.RequestError:
            st.write("Sorry, the service is down.")
            return None

# Function to display sign video
def display_sign_video(sign_text):
    video_path = os.path.join(DATA_FOLDER, f"{sign_text}.mp4")
    if os.path.exists(video_path):
        return video_path
    else:
        st.write(f"No video found for '{sign_text}'")
        return None

# Custom WebRTC video processor for real-time video processing
class SignLanguageProcessor(VideoProcessorBase):
    def __init__(self):
        self.frames_list = []
        self.start_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_resized = cv2.resize(img, (IMAGE_HEIGHT, IMAGE_WIDTH))
        img_normalized = img_resized / 255
        self.frames_list.append(img_normalized)

        elapsed_time = time.time() - self.start_time

        if elapsed_time >= 4 and len(self.frames_list) >= SEQUENCE_LENGTH:
            predicted_class_name, confidence = predict_single_action(self.frames_list[-SEQUENCE_LENGTH:])
            st.session_state.prediction = f"Predicted: {predicted_class_name} (Confidence: {confidence:.2f})"
            speak(predicted_class_name)
            self.start_time = time.time()
            self.frames_list = []

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app
st.title("Live Video Action Recognition")

# Mode selection
mode = st.radio("Select Mode", ("Sign to Speech", "Speech to Sign"))

if mode == "Sign to Speech":
    st.write("Starting real-time sign language recognition...")
    
    # Initialize WebRTC Stream
    ctx = webrtc_streamer(
        key="sign-recognition",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=SignLanguageProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    if "prediction" not in st.session_state:
        st.session_state.prediction = "No prediction yet."
    st.write(st.session_state.prediction)

elif mode == "Speech to Sign":
    st.write("Click 'Start' and say a phrase.")
    
    if st.button("Start"):
        text = recognize_speech()
        if text:
            video_path = display_sign_video(text)
            if video_path:
                st.video(video_path, format="video/mp4", loop=True)
