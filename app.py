import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from PIL import Image
file_id = "1v64ACt3QX7X1Q_avc02LTfZpLR9T7h7e"
url = 'https://drive.google.com/uc?id=1v64ACt3QX7X1Q_avc02LTfZpLR9T7h7e'
model_path = "trained_plant_disease_model.keras"
# Constants
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = (128, 128)

# Cache the model loading using Streamlit's cache_resource decorator
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model.keras")

def model_prediction(test_image):
    # Load image using PIL and preprocess
    img = Image.open(test_image)
    img = img.resize(IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    
    # Make prediction
    model = load_model()
    predictions = model.predict(img_array)
    return np.argmax(predictions)

# Sidebar configuration
st.sidebar.title("Plant Disease System For Sustainable Agriculture")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

# Page routing
if app_mode == 'Home':
    st.markdown("<h1 style='text-align:center;'>Plant Disease Detection System For Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)
    img = Image.open('Disease.png')
    st.image(img, use_container_width=True)
    
elif app_mode == 'Disease Recognition':
    st.header('Plant Disease Detection')
    st.subheader('Upload a plant leaf image for disease detection')
    
    # File uploader with supported formats
    test_image = st.file_uploader('Choose an image:', type=['jpg', 'jpeg', 'png'],help="Supported formats: JPG, JPEG, PNG")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button('Show Image') and test_image:
            st.image(test_image, caption='Uploaded Image', use_container_width=True)
    
    with col2:
        if st.button('Predict', key="predict_btn"):
            if not test_image:
                st.warning("Please upload an image before predicting.")
            else:
                with st.spinner('Analyzing...'):
                    try:
                        result_index = model_prediction(test_image)
                        st.success(f'Prediction: {CLASS_NAMES[result_index]}')
                        st.snow()
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")

            
