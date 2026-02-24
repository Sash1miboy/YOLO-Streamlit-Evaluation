import cv2
import streamlit as st

from utils.load_data import load_yolo_model


def demo_page(model_name: str):

    st.title("Live YOLO Demo (CPU)")
    st.text(f"Model used: {model_name}")

    model = load_yolo_model(model_name)
    
    if "run_demo" not in st.session_state:
        st.session_state.run_demo = False

    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start DEMO", width="stretch"):
            st.session_state.run_demo = True
    with c2:
        if st.button("⏹️ Stop DEMO", width="stretch"):
            st.session_state.run_demo = False

    run_demo = st.session_state.run_demo

    live_frame_window = st.image([], width="stretch")

    if run_demo:
        camera = cv2.VideoCapture(0)

        while st.session_state.run_demo:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            results = model(frame, imgsz=736, conf=0.25)
            annotated_frame = results[0].plot()

            live_frame_window.image(annotated_frame, channels="BGR", width="stretch")

        camera.release()
