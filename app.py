import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('rice_classifier.h5')
labels = ['Basmati', 'Jasmine', 'Arborio', 'Brown', 'Parboiled']

st.title("Rice Type Identifier")
uploaded = st.file_uploader("Upload an image of rice grain")

if uploaded:
    img = image.load_img(uploaded, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    st.write(f"Predicted Rice Type: {labels[np.argmax(prediction)]}")
