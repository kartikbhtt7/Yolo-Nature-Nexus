import streamlit as st
import cv2
import numpy as np
import tempfile
from utils.onnx_inference import YOLOv11

st.set_page_config(page_title="Forest Surveillance", layout="wide")
st.title("Forest Surveillance System")

@st.cache_resource
def load_model():
    return YOLOv11('models/best_model.onnx')

model = load_model()

# Sidebar controls
st.sidebar.header("Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
iou_thres = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5)
model.conf_thres = confidence
model.iou_thres = iou_thres

# Main interface
tab1, tab2, tab3 = st.tabs(["Image", "Video", "Webcam"])

with tab1:
    img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if image is not None:
            detections = model.detect(image)
            result_image = model.draw_detections(image.copy(), detections)
            st.image(result_image, channels="BGR", use_container_width=True)

with tab2:
    video_file = st.file_uploader("Upload Video", type=["mp4", "avi"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        output_path = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, 
                            (int(cap.get(3)), int(cap.get(4))))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            detections = model.detect(frame)
            result_frame = model.draw_detections(frame, detections)
            out.write(result_frame)
            stframe.image(result_frame, channels="BGR")
        
        cap.release()
        out.release()
        st.success("Processing completed!")
        st.download_button("Download Video", open(output_path, 'rb'), "processed.mp4")

with tab3:
    cam = st.camera_input("Live Camera Feed")
    if cam:
        image = cv2.imdecode(np.frombuffer(cam.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            detections = model.detect(image)
            result_image = model.draw_detections(image.copy(), detections)
            st.image(result_image, channels="BGR")