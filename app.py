import streamlit as st
import cv2
import time
import numpy as np
import pyttsx3
import speech_recognition as sr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Dropout, Flatten, LSTM, Dense
import os

st.set_page_config(layout="wide", page_title="Sign Language AI")

if 'text_history' not in st.session_state:
    st.session_state['text_history'] = []

SEQUENCE_LENGTH = 30
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["Namaste", "Good Afternoon", "Good Night", "How are you"]
DATA_FOLDER = "data"

@st.cache_resource
def load_model():
    try:
        model = Sequential()
        model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='relu'),
                                  input_shape=(SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
        model.add(TimeDistributed(MaxPooling2D((4, 4))))
        model.add(TimeDistributed(Dropout(0.25)))
        model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((4, 4))))
        model.add(TimeDistributed(Dropout(0.25)))
        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Dropout(0.25)))
        model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu')))
        model.add(TimeDistributed(MaxPooling2D((2, 2))))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(32))
        model.add(Dense(len(CLASSES_LIST), activation='softmax'))
        
        if os.path.exists("LRCN_model.h5"):
            model.load_weights("LRCN_model.h5")
            return model
        else:
            return None
    except Exception as e:
        return None

model = load_model()

def predict_action(frames_list, model):
    frames_array = np.expand_dims(frames_list, axis=0)
    probabilities = model.predict(frames_array)[0]
    label_index = np.argmax(probabilities)
    return CLASSES_LIST[label_index], probabilities[label_index]

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

def listen_mic():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.listen(source, timeout=3, phrase_time_limit=3)
            text = r.recognize_google(audio)
            return text
        except:
            return None

st.title("AI Sign Language System")

if model is None:
    st.error("❌ Error: 'LRCN_model.h5' not found. Please place the file in the script directory.")
    st.stop()

with st.sidebar:
    st.header("Control Panel")
    mode = st.radio("Select Mode", ["Sign to Speech", "Speech to Sign"])
    st.divider()
    st.info("Note: Use the toggle switches below to Start/Stop the system cleanly.")

if mode == "Sign to Speech":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Feed")
        cam_placeholder = st.empty()
        
    with col2:
        st.subheader("Controls")
        run_camera = st.toggle("Turn Camera On/Off", value=False)
        
        st.subheader("Prediction")
        action_display = st.empty()
        conf_display = st.empty()
        
    if run_camera:
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open camera. Check permissions or try a different USB port.")
        else:
            frames_list = []
            start_time = time.time()
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Lost connection to camera.")
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cam_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                resized = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
                normalized = resized / 255
                frames_list.append(normalized)
                
                elapsed = time.time() - start_time
                if elapsed > 4:
                    if len(frames_list) >= SEQUENCE_LENGTH:
                        action, conf = predict_action(frames_list[-SEQUENCE_LENGTH:], model)
                        
                        action_display.metric("Action", action)
                        conf_display.metric("Confidence", f"{conf:.2f}")
                        
                        speak(f"Predicted {action}")
                        
                    frames_list = []
                    start_time = time.time()
            
            cap.release()
    else:
        cam_placeholder.info("Camera is OFF. Toggle the switch to start.")

elif mode == "Speech to Sign":
    VIDEO_MAP = {
        "namaste": "Namaste",
        "good afternoon": "Good Afternoon",
        "good night": "Good Night",
        "how are you": "How are you"
    }

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Microphone")
        listen_active = st.toggle("Activate Microphone", value=False)
        st.write("Say: *Namaste, Good Afternoon, Good Night, How are you*")
        status_txt = st.empty()
        
    with col2:
        st.subheader("Avatar")
        video_placeholder = st.empty()
        
    if listen_active:
        status_txt.info("Listening... (Speak now)")
        
        while listen_active:
            text = listen_mic()
            
            if text:
                status_txt.success(f"Heard: '{text}'")
                
                clean_text = text.lower().strip().replace(".", "")
                
                if clean_text in VIDEO_MAP:
                    filename = VIDEO_MAP[clean_text]
                    video_path = os.path.join(DATA_FOLDER, f"{filename}.mp4")
                    
                    if os.path.exists(video_path):
                        video_placeholder.video(video_path, format="video/mp4", loop=True, autoplay=True)
                    else:
                        st.warning(f"Video file '{filename}.mp4' missing in data folder.")
                else:
                    st.warning("Phrase not recognized in database.")
            else:
                status_txt.info("Listening...")
                
            time.sleep(0.1)