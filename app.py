import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from PIL import Image
import os

# Constants
MODEL_FILE_ID = "1v64ACt3QX7X1Q_avc02LTfZpLR9T7h7e"
MODEL_PATH = "trained_plant_disease_model.keras"
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = (128, 128)

# Download model only if not exists
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f'https://drive.google.com/uc?id={MODEL_FILE_ID}', MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

# Cache model loading
@st.cache_resource
def get_model():
    return load_model()

# Image preprocessing
def preprocess_image(image_path, target_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    return np.expand_dims(input_arr, axis=0)

# Prediction function
def model_prediction(test_image):
    model = get_model()
    processed_image = preprocess_image(test_image, IMAGE_SIZE)
    predictions = model.predict(processed_image)
    return np.argmax(predictions)

# Configure page
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌱")

# Sidebar navigation
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox('Select Page', ['Home', 'Disease Recognition'])

# Page handling
if app_mode == 'Home':
    st.title("Plant Disease Detection System for Sustainable Agriculture 🌱")
    img = Image.open('Disease.png')
    st.image(img, use_container_width=True)
    st.markdown("""
        ### Welcome!
        This system helps identify plant diseases in potato crops using AI.
        - **Home**: General information
        - **Disease Recognition**: Upload plant images for disease detection
    """)

elif app_mode == 'Disease Recognition':
    st.title("Plant Disease Detection")
    st.subheader("Identify Potato Plant Diseases")
    
    uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Temporary file handling
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                try:
                    prediction_idx = model_prediction(temp_file)
                    disease = CLASS_NAMES[prediction_idx]
                    status = "Healthy" if "healthy" in disease.lower() else "Diseased"
                    
                    with col2:
                        st.success("Analysis Complete!")
                        st.subheader("Results")
                        st.metric("Status", status)
                        st.metric("Prediction", disease.split("___")[-1].replace("_", " "))
                        st.balloons()
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                finally:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

# Add footer
st.markdown("---")
st.markdown("### Sustainable Agriculture Initiative 2024 | [Learn More](#)")