import streamlit as st
from PIL import Image, ImageDraw
import plotly.express as px
import numpy as np
from ultralytics import YOLO
import io

model_name = "yolov8l400"
model_path = f"../../generated/{model_name}/weights/best.pt"

# @st.experimental_singleton
def load_model(model_path):
    return YOLO(model_path)

model = load_model(model_path)

def main():
    st.title("bone fracture detection")

    images = st.file_uploader(label="Upload image", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    annotation_file = st.file_uploader(label="Upload annotation (optional)", type=["txt"])

    threshold = st.slider(label="", min_value=0.001, max_value=0.5, value=0.1, step=0.001, format="%0.3f")

    image_results = []

    with st.spinner("Detecting fractures..."):
        # create image placeholders
        px_image_figures = []
        st_image_elements = []
        # st_text_elements = []
        for _ in range(len(images)):
            # st_text_elements.append(st.empty())
            st_image_elements.append(st.empty())

        for i in range(len(images)):
            images[i] = open_image(images[i])

            fig = imshow(images[i])
            px_image_figures.append(fig)
            st_image_elements[i].plotly_chart(fig, key=f"plotly_chart_initial_{i}")

        for i in range(len(images)):
            result = predict(images[i])

            new_result = []
            for i, r in enumerate(result):
                if r["confidence"] >= threshold:
                    new_result.append(r)
            result = new_result

            image_results.append(result)

    for i in range(len(images)):
        fig = px_image_figures[i]

        if annotation_file is not None:
            img_size_x, img_size_y = images[i].size
            add_annotation_file_boxes(fig, annotation_file, img_size_x, img_size_y)

        if len(image_results[i]) > 0:
            add_model_prediction_boxes(fig, image_results[i])

        st_image_elements[i].plotly_chart(fig, key=f"plotly_chart_annotated_{i}")


        if len(image_results[i]) > 0:
            annotated_image = create_downloadable_image(images[i], result)
            img_buffer = io.BytesIO()
            annotated_image.save(img_buffer, format='PNG')
            img_bytes = img_buffer.getvalue()

            st.download_button(
                label="Download Annotated Image",
                data=img_bytes,
                file_name=f"annotated_{images[i].filename if hasattr(images[i], 'filename') else 'image'}.png",
                mime="image/png"
            )

# @st.experimental_memo
def open_image(image):
    image = Image.open(image)
    image = image.convert("RGBA")
    return image

# @st.experimental_memo
def imshow(_image):
    return px.imshow(_image)

# @st.experimental_memo
def predict(_image):
    return model.predict(_image, conf=0.001)[0].summary()

def add_annotation_file_boxes(fig, annotation_file, img_size_x, img_size_y):
    annotation = annotation_file.read().decode("utf-8")
    boxes = read_bounding_boxes(annotation)

    for b in boxes:
        x0, y0, x1, y1 = b[0] * img_size_x, b[1] * img_size_y, b[2] * img_size_x, b[3] * img_size_y
        fig.add_shape(x0=x0, y0=y0, x1=x1, y1=y1, line={"color": "rgba(0, 0, 255, 0.4)"})

def add_model_prediction_boxes(fig, result):
    for rect in result:
        fig.add_shape(x0=rect["box"]["x1"], y0=rect["box"]["y1"], x1=rect["box"]["x2"], y1=rect["box"]["y2"],
                      line={"color": "rgba(255, 0, 0, 0.4)"})


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

def create_downloadable_image(image, results):
    annotated_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated_image)

    for rect in results:
        x0, y0, x1, y1 = rect["box"]["x1"], rect["box"]["y1"], rect["box"]["x2"], rect["box"]["y2"]
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=3)
        draw.text((x0, y0-15), f"{rect['confidence']:.3f}", fill=(255, 0, 0))

    return annotated_image

if __name__ == "__main__":
    main()
