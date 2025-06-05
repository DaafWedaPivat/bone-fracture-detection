import streamlit as st
from PIL import Image, ImageDraw


def main():
    st.title("bone fracture detection")

    model_name = "yolov8n100"
    model_path = f"../generated/{model_name}/weights/best.pt"

    image = st.file_uploader(label="Upload image", type=["png", "jpg", "jpeg"])

    annotation_file = st.file_uploader(label="Upload annotation (optional)", type=["txt"])


    if image is not None:
        from ultralytics import YOLO

        image = Image.open(image)

        image = image.convert("RGBA")

        # st.image(image, use_column_width=True)
        model = YOLO(model_path)
        result = model.predict(image, conf=0.01)[0]

        imdraw = ImageDraw.Draw(image)

        if annotation_file is not None:
            img_size_x, img_size_y = image.size
            annotation = annotation_file.read().decode("utf-8")
            boxes = read_bounding_boxes(annotation)
            # annotation = annotation.split(" ")[1:]
            # if len(annotation) >= 4:
            #     annotation = [float(e) for e in annotation]
            #
            #
            #     abs_annotation = [annotation[0] - annotation[2]/2, annotation[1] - annotation[3]/2, annotation[0] + annotation[2]/2, annotation[1] + annotation[3]/2]
            #     abs_annotation = [abs_annotation[0]*img_size_x, abs_annotation[1]*img_size_y, abs_annotation[2]*img_size_x, abs_annotation[3]*img_size_y]
            for b in boxes:
                b = [b[0]*img_size_x, b[1]*img_size_y, b[2]*img_size_x, b[3]*img_size_y]
                imdraw.rectangle(b, outline=(0, 0, 255, 255), width=3)

        if len(result.boxes.cls) > 0:
            "Bruch gefunden"

            for rect in result.boxes.xyxy.tolist():
                imdraw.rectangle(rect, outline=(255, 0, 0, 100), width=3)

            st.image(image, use_column_width=True)

        else:
            "kein Bruch gefunden"
            st.image(image, use_column_width=True)


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
