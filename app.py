# app.py
import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from preprocess import preprocess_frame

model = load_model("indian_model")  
labels = sorted(os.listdir("Indian")) 
L_index = labels.index('L')
penalty_factor = 0.5

st.title("ISL Hand Sign Recognition (Image Upload)")
st.write("Upload an image of a hand sign (A-Z or 0-9).")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

    processed = preprocess_frame(image)
    st.image(processed[0], caption="Preprocessed Image", clamp=True)
    prediction = model.predict(processed, verbose=0)
    prediction[0][L_index] *= penalty_factor
    class_index = np.argmax(prediction)
    predicted_char = labels[class_index]

    st.success(f"Predicted Sign: **{predicted_char}**")
