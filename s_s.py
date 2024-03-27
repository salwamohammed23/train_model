# Import the Libraries
import cv2
import pandas as pd
from ultralytics import YOLO
import streamlit as st
from io import BytesIO
import numpy as np



model=YOLO('model_- 25 march 2024 2_53.pt')


def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5):
    img_copy = img.copy()  # Create a copy of the original image
    results = predict(chosen_model, img_copy, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img_copy, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
            cv2.putText(img_copy, f"{result.names[int(box.cls[0])]}",
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)
    return img_copy, results





st.title("Upload the Image for Detections!!")

# Upload image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])



if uploaded_file is not None:
    # Read image file
    image_bytes = uploaded_file.getvalue()
    orig_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Perform object detection
    result_img = predict_and_detect(model, orig_image, classes=[], conf=0.5)

    # Display the original image
    st.subheader("Original Image")
    st.image(orig_image, caption='Original Image', use_column_width=True)

    # Display the detected image
    st.subheader("Detected Objects")
    st.image(result_img[0], caption='Detected Objects', use_column_width=True)
