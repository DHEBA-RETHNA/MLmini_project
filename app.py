#app.py
import gradio as gr
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from preprocess import preprocess_frame

model = load_model("indian_model.h5")
labels = sorted(os.listdir("Indian"))
L_index = labels.index('L')
penalty_factor = 0.5

def predict_hand_sign(image):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    processed = preprocess_frame(image_bgr)
    prediction = model.predict(processed, verbose=0)
    prediction[0][L_index] *= penalty_factor
    class_index = np.argmax(prediction)
    predicted_char = labels[class_index]

    display_image = (processed[0, :, :, 0] * 255).astype(np.uint8)
    
    return display_image, predicted_char

iface = gr.Interface(
    fn=predict_hand_sign,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(type="numpy", label="Preprocessed Image"), gr.Textbox(label="Predicted Sign")],
    title="ISL Hand Sign Recognition (Image Upload)",
    description="Upload an image of a hand sign (A-Z or 0-9)."
)

if __name__ == "__main__":
    iface.launch(share=True)
