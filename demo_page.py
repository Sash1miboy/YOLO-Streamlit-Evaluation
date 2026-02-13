import streamlit as st
import cv2
from ultralytics import YOLO

def demo_page(model_name: str):

    st.title("Live FER Demo")
    st.text("Model used : " + model_name)
    
    @st.cache_resource
    def load_model(model_name):
        return YOLO(f"models/{model_name}")

    model = load_model(model_name)


    if "run" not in st.session_state:
        st.session_state.run = False

    if st.button("🎥 Start / Stop Webcam", use_container_width=True):
        st.session_state.run = not st.session_state.run

    run = st.session_state.run

    FRAME_WINDOW = st.image([], use_container_width=True)

    if run:
        camera = cv2.VideoCapture(0)

        while st.session_state.run:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            FRAME_WINDOW.image(
                annotated_frame,
                channels="BGR",
                use_container_width=True
            )

        camera.release()
