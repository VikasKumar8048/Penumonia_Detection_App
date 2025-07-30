import base64
import streamlit as st
from PIL import ImageOps, Image
import numpy as np

def set_background(image_file):
    """
    Sets a background image for the Streamlit app.

    Parameters:
        image_file (str): Path to the background image file.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    """
    Classifies the input image using the provided trained model.

    Parameters:
        image (PIL.Image.Image): Input image to classify.
        model (keras.Model): Trained Keras model for classification.
        class_names (list): List of class names corresponding to model output.

    Returns:
        tuple: (Predicted class name, Confidence score)
    """
    # Resize image to 224x224
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # Convert image to numpy array and normalize
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1

    # Prepare model input shape
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction[0])
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score
