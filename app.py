import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image, ExifTags

# --------------------- App Configuration ---------------------
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

# --------------------- Download & Load Model Efficiently ---------------------
file_id = "1v64ACt3QX7X1Q_avc02LTfZpLR9T7h7e"
url = 'https://drive.google.com/uc?id=1v64ACt3QX7X1Q_avc02LTfZpLR9T7h7e'
model_path = "trained_plant_disease_model.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        with st.spinner("🔄 Downloading model... Please wait!"):
            gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()

# --------------------- Image Preprocessing Function ---------------------
def preprocess_image(image):
    try:
        # Fix image orientation
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = dict(image._getexif().items())
            if orientation in exif:
                if exif[orientation] == 3:
                    image = image.rotate(180)
                elif exif[orientation] == 6:
                    image = image.rotate(270)
                elif exif[orientation] == 8:
                    image = image.rotate(90)

        # Convert to RGB and Resize
        image = image.convert("RGB")
        image = image.resize((128, 128))

        # Convert to array and normalize
        img_array = np.array(image) / 255.0  # Normalize pixels (0-1)
        return np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# --------------------- Disease Prediction Function ---------------------
def model_prediction(image):
    processed_image = preprocess_image(image)
    if processed_image is None:
        return None
    predictions = model.predict(processed_image)
    return np.argmax(predictions)  # Return index of max probability class

# --------------------- Sidebar Navigation ---------------------
st.sidebar.title("🌱 Plant Disease Detector")
app_mode = st.sidebar.radio("📌 Choose an option:", ["🏠 Home", "🔍 Disease Detection"])

# --------------------- Home Page ---------------------
if app_mode == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:#2E8B57;'>🌿 Plant Disease Detection System 🌿</h1>", unsafe_allow_html=True)
    st.image("Disease.png", use_container_width=True)
    
    st.write("""
    🌾 **Welcome to the Plant Disease Detection System!**  
    This tool uses **Deep Learning** to detect plant diseases, even if the image is taken from **any direction**!  
    📌 **How to Use?**  
    - Click on **Disease Detection** from the sidebar.  
    - Upload or take a **leaf image** for analysis.  
    - Get an **instant prediction** on its health!  
    """)
    
    st.markdown("---")
    st.info("Go to **Disease Detection** to start predicting!")

# --------------------- Disease Detection Page ---------------------
elif app_mode == "🔍 Disease Detection":
    st.markdown("<h2 style='text-align:center; color:#2E8B57;'>🔬 Plant Disease Detection</h2>", unsafe_allow_html=True)

    # File Uploader for Image
    test_image = st.file_uploader("📸 Upload or Take a Picture of a Plant Leaf", type=["jpg", "png", "jpeg"])

    # Show Image Button
    if test_image:
        if st.button("🖼 Show Image"):
            st.image(test_image, caption="📷 Uploaded Image", use_container_width=True)
            st.success("✅ Image Displayed Successfully!")

        # Prediction Button
        if st.button("🚀 Predict Now"):
            with st.spinner("🧠 AI is analyzing the image..."):
                result_index = model_prediction(Image.open(test_image))

                if result_index is not None:
                    class_names = ['Potato - Early Blight', 'Potato - Late Blight', 'Potato - Healthy']
                    prediction = class_names[result_index]

                    # Display Prediction Result
                    st.success(f"🎯 Model Prediction: **{prediction}**")
                    st.snow()  # 🎊 Confetti Effect Instead of Balloons
                else:
                    st.error("❌ Failed to process the image. Please try again.")
