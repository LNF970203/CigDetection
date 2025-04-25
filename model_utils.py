from ultralytics import YOLO
import streamlit as st


@st.cache_resource
def load_yolo_model(model_path):

    return YOLO(model_path)