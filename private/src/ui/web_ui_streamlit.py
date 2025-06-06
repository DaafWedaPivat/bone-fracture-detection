import streamlit as st
from PIL import Image, ImageDraw
import plotly.express as px
import numpy as np
from ultralytics import YOLO

model_name = "yolo11m400"
model_path = f"../../generated/{model_name}/weights/best.pt"
model = YOLO(model_path)

def main():
    st.title("bone fracture detection")

    images = st.file_uploader(label="Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    annotation_file = st.file_uploader(label="Upload annotation (optional)", type=["txt"])

    threshold = st.slider(label="Confidence threshold", min_value=0.001, max_value=0.5, value=0.05, step=0.001, format="%0.3f")

    for image in images:
        image = Image.open(image)

        image = image.convert("RGBA")

    with st.spinner("Detecting fractures..."):
        result = model.predict(image, conf=0.001)[0].summary()

        new_result = []
        for i, r in enumerate(result):
            if r["confidence"] >= threshold:
                new_result.append(r)
        result = new_result

        fig = px.imshow(image)

        if annotation_file is not None:
            img_size_x, img_size_y = image.size
            annotation = annotation_file.read().decode("utf-8")
            boxes = read_bounding_boxes(annotation)

            for b in boxes:
                x0, y0, x1, y1 = b[0]*img_size_x, b[1]*img_size_y, b[2]*img_size_x, b[3]*img_size_y
                fig.add_shape(x0=x0, y0=y0, x1=x1, y1=y1, line={"color":"rgba(0, 0, 255, 0.4)"})

        if len(result) > 0:
            "Bruch gefunden"

            for rect in result:
                # imdraw.rectangle(rect, outline=(255, 0, 0, 100), width=3)
                fig.add_shape(x0=rect["box"]["x1"], y0=rect["box"]["y1"], x1=rect["box"]["x2"], y1=rect["box"]["y2"], line={"color":"rgba(255, 0, 0, 0.4)"})

            # st.image(image, use_column_width=True)

        else:
            "kein Bruch gefunden"
            # st.image(image, use_column_width=True)
        st.plotly_chart(fig, use_container_width=True)


def read_bounding_boxes(string):
    bounding_boxes = []

    for line in string.splitlines():
        line = line.strip()
        if len(line) > 0:  # Check if the line is not empty
            # Split the line into components and convert to float
            x_center, y_center, w, h = map(float, line.split()[1:])

            # Calculate the coordinates of the bounding box
            x1 = x_center - (w / 2)
            y1 = y_center - (h / 2)
            x2 = x_center + (w / 2)
            y2 = y_center + (h / 2)

            # Append the bounding box to the list
            bounding_boxes.append([x1, y1, x2, y2])

    return bounding_boxes


if __name__ == "__main__":
    main()
