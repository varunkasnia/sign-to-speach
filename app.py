import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os

# Constants
SEQUENCE_LENGTH = 30  # Number of frames per sequence
CLASSES_LIST = ["Namaste", "Good Afternoon", "Good Night", "How are you"]  # Replace with your class names
DATA_FOLDER = "data"  # Folder containing sign videos

# Load your pre-trained LRCN model
LRCN_model = load_model("LRCN_model.h5")  # Replace with your model path

# Function to predict action
def predict_action():
    ''' Simulates action recognition without video input '''
    random_probabilities = np.random.rand(len(CLASSES_LIST))
    predicted_label = np.argmax(random_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    return predicted_class_name, random_probabilities[predicted_label]

# Speech-to-text simulation
def recognize_speech():
    return st.text_input("Enter speech input:", "Hello")

# Function to display sign simulation
def display_sign_video(sign_text):
    if sign_text in CLASSES_LIST:
        return f"Simulated sign display: {sign_text}"
    else:
        return "No matching sign found."

# Streamlit app
st.title("Text-Based Sign to Speech & Speech to Sign")

# Mode selection
mode = st.radio("Select Mode", ("Sign to Speech", "Speech to Sign"))

if mode == "Sign to Speech":
    st.write("Click 'Predict' to simulate sign recognition.")
    if st.button("Predict"):
        predicted_class_name, confidence = predict_action()
        st.write(f"Simulated Prediction: {predicted_class_name} (Confidence: {confidence:.2f})")

elif mode == "Speech to Sign":
    st.write("Enter text to simulate speech input.")
    text = recognize_speech()
    if text:
        sign_result = display_sign_video(text)
        st.write(sign_result)
