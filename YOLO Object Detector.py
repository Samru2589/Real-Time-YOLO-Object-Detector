import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import torch

# Load YOLO model with GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO('yolov8n.pt').to(device)

# Define a dictionary with class names as keys and colors as values
COLORS = {
    'person': (0, 255, 255),      # Yellow
    'car': (255, 0, 0),           # Blue
    'bus': (0, 255, 0),           # Green
    'truck': (0, 0, 255),         # Red
    'book': (255, 255, 0),        # Light Blue
    'pen': (255, 0, 255),         # Purple
    'chair': (255, 165, 0),       # Orange
    'bottle': (0, 255, 255),      # Cyan
    'cup': (255, 192, 203),       # Pink
    'cell phone': (255, 255, 0),  # Light Yellow
    'laptop': (0, 255, 255),      # Light Blue (new)
    'cat': (128, 0, 128),         # Purple (new)
    'dog': (0, 128, 0)            # Dark Green (new)
    # Add more classes and colors as needed
}

def process_image_with_yolo(model, image):
    # Resize the image for faster processing
    original_shape = image.shape
    image_resized = cv2.resize(image, (640, 640))

    # Perform inference
    results = model(image_resized)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = box.cls[0].item()
            class_name = result.names[int(cls)]

            # Scale coordinates back to the original image size
            x1 = int(x1 * original_shape[1] / 640)
            x2 = int(x2 * original_shape[1] / 640)
            y1 = int(y1 * original_shape[0] / 640)
            y2 = int(y2 * original_shape[0] / 640)

            # Get color for the class
            color = COLORS.get(class_name, (0, 255, 255))  # Default to yellow if class not in COLORS

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label and confidence
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def main():
    st.title("YOLOv8 Object Detection")

    # Sidebar for uploading files
    st.sidebar.title("Upload Options")
    upload_option = st.sidebar.selectbox("Choose an option", ["Image", "Video", "Webcam"])

    if upload_option == "Image":
        uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png","jfif"])

        if uploaded_image is not None:
            image = np.array(Image.open(uploaded_image))
            processed_image = process_image_with_yolo(model, image)
            st.image(processed_image, caption="Processed Image", use_column_width=True)

    elif upload_option == "Video":
        uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

        if uploaded_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = process_image_with_yolo(model, frame)
                stframe.image(frame, channels="BGR", use_column_width=True)

            cap.release()

    elif upload_option == "Webcam":
        run_webcam_detection()

def run_webcam_detection():
    st.header("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture video")
            break

        frame = process_image_with_yolo(model, frame)
        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
