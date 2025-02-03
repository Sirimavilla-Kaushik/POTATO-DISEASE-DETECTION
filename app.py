import streamlit as st 
import tensorflow as tf
import numpy as np
import gdown
import os
file_id="1v64ACt3QX7X1Q_avc02LTfZpLR9T7h7e"
url='https://drive.google.com/uc?id=1v64ACt3QX7X1Q_avc02LTfZpLR9T7h7e'
model_path="trained_plant_disease_model.keras"

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url,model_path,quiet=False)

def model_prediction(test_image):
    model=tf.keras.models.load_model("trained_plant_disease_model.keras")
    image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr=tf.keras.preprocessing.image.img_to_array(image)
    input_arr=np.array([input_arr])#convert single image to batch
    predictions=model.predict(input_arr)
    return np.argmax(predictions)#return index of max element

#sidebar
st.sidebar.title("Plant Disease system for Sustainable Agriculture")
app_mode=st.sidebar.selectbox('select page',['Home','Disease Recognition'])

from PIL import Image
img=Image.open('Disease.png')
st.image(img)

if(app_mode=='HOME'):
    st.markdown("<h1 style='text-align:center;'>Plant Disease Detection system for Sustainable Architecture",unsafe_allow_html=True)
    
elif(app_mode=='Disease Recognition'):
    st.header('Plant Disease Detection System For Sustainable Agriculture')

test_image=st.file_uploader('choose an image:')
if(st.button('Show Image')):
    st.image(test_image,width=4,use_container_width=True)

if(st.button('Predict')):
    st.snow()
    st.write('our prediction')
    result_index=model_prediction(test_image)
    class_name=['Potato___Early_blight','Potato___Late_blight','Potato___healthy']
    st.success('Model is predicting its a {}'.format(class_name[result_index]))
    