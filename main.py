import streamlit as st
# from keras.models import load_model
from keras.models import load_model

from PIL import Image
import numpy as np
from util import classify, set_background

# Set background image
set_background('good_doctor.jpg')

# Title and header with custom color
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>Pneumonia Classification</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='color: #117A65;'>Please upload a chest X-ray image</h3>",
    unsafe_allow_html=True
)

# File uploader with accessible label
file = st.file_uploader('Upload Image', type=['jpeg', 'jpg', 'png'], label_visibility='collapsed')

# Load model without compile warning
model = load_model('keras_model.h5', compile=False)

# Load class names
with open('labels.txt', 'r') as f:
    class_names = [line.strip().split(' ')[1] for line in f.readlines()]

# If image uploaded
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify the image
    class_name, conf_score = classify(image, model, class_names)

    # Display colored prediction result
    st.markdown(
        f"<h2 style='color: #D35400;'>🩺 Prediction: {class_name}</h2>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<h4 style='color: #7D3C98;'>Confidence Score: {round(conf_score * 100, 2)}%</h4>",
        unsafe_allow_html=True
    )
