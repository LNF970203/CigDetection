import streamlit as st
import io
from PIL import Image
import os
import shutil

from model_utils import load_yolo_model
from image_ai import compare_images


USER_IMAGE_NAME = "user_input.jpg"
PREDICTION_PATH = "runs/detect/predict"
PREDICTION_NEW_PATH = "predictions"
PREDICTION_IMAGE_PATH = PREDICTION_NEW_PATH + "/" + USER_IMAGE_NAME
MODEL_NAME = "best.pt"
REFERENCE_IMAGE = "reference.png"


#load the pytorch weights
model = load_yolo_model(MODEL_NAME)


#get predictions
def get_predictions(source_path):
    model.predict(source_path, save = True)
    print("Prediction Complete")

    #copy the predictions and save it independently
    #this is mainly to avoid prediction tree structure
    if os.path.exists(PREDICTION_PATH):
        for item in os.listdir(PREDICTION_PATH):
            shutil.copy(os.path.join(PREDICTION_PATH, item), os.path.join(PREDICTION_NEW_PATH, item))
            os.remove(os.path.join(PREDICTION_PATH, item))

    #then remove the predict director
    os.rmdir(PREDICTION_PATH)
    print("Folder removed!")

    return True


# web app
st.title("CigPanel Detection")

tab_one, tab_two = st.tabs(["File Uploader", "Camera Input"])

with tab_one:
    file_image = st.file_uploader("Upload your file")
    # predictions
    if file_image:
        # columns for images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("User Input")
            #set the user image
            st.image(file_image)

            #read and save the image
            image_bytes = io.BytesIO(file_image.read())
            input_image = Image.open(image_bytes)
            input_image.save(USER_IMAGE_NAME)

        # get predictions
        results = get_predictions(USER_IMAGE_NAME)

        if results:
            with col2:
                st.subheader("Prediction")
                #set the user image
                st.image(PREDICTION_IMAGE_PATH)

        if results:
            with st.spinner("Comparing...", show_time=True):
                response = compare_images(REFERENCE_IMAGE, PREDICTION_IMAGE_PATH)
                st.write(response)    
                


with tab_two:
    cam_image = st.camera_input("Take Photo")
    # predictions
    if cam_image:
        # columns for images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("User Input")
            #set the user image
            st.image(cam_image)

            #read and save the image
            image_bytes = io.BytesIO(cam_image.read())
            input_image = Image.open(image_bytes)
            input_image.save(USER_IMAGE_NAME)

        # get predictions
        results = get_predictions(USER_IMAGE_NAME)

        if results:
            with col2:
                st.subheader("Prediction")
                #set the user image
                st.image(PREDICTION_IMAGE_PATH)

        if results:
            with st.spinner("Comparing...", show_time=True):
                response = compare_images(REFERENCE_IMAGE, PREDICTION_IMAGE_PATH)
                st.write(response)  

