import os
from PIL import Image

from sklearn.metrics import roc_curve, RocCurveDisplay
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
print("Import ready")

iou_threshold = 0.5
batch_size = 32


def main():
    models = [
              "yolo11n100",
              "yolo11n100_enhanced",
              ]

    test_data = "../generated/yolo_dataset/valid/"
    images = [test_data + "images/" + i for i in os.listdir(test_data + "images")]

    images = check_images(images)

    test_data_enhanced = "../generated/yolo_dataset_enhanced/valid/"
    images_enhanced = [test_data_enhanced + "images/" + i for i in os.listdir(test_data_enhanced + "images")]

    images_enhanced = check_images(images_enhanced)
    print("check images ready")

    # ax = plt.axes()
    # x = np.linspace(0, 1, 100)
    # ax.plot(x, x, linestyle=":")
    fig = go.Figure()

    for model in models:
        model_path = f"../generated/{model}/weights/best.pt"
        print(model_path)
        if "enhanced" in model:
            print("enhanced model")
            true_labels, confidence_values = evaluate_model(model_path, images_enhanced, test_data_enhanced + "labels/")
        else:
            true_labels, confidence_values = evaluate_model(model_path, images, test_data + "labels/")
        # plot_roc_curve(ax, true_labels, confidence_values, name=model)
        plot_roc_curve_detailed(fig, true_labels, confidence_values, name=model)
        # break
        fig.update()

    output_path = "../generated/roc_curve.html"
    fig.write_html(output_path)
    print(f"ROC curve saved to {output_path}")
    fig.show()


def evaluate_model(model_path, images, label_directory) -> (list, list):
    """
    evaluates a model to generate ROC curve
    :param model_path: path to model.pt
    :param images: list of paths to images
    :return: (true_labels, confidence_values)
    """
    model = YOLO(model_path)
    print("load model ready")

    results = []

    count = 0
    while count < len(images):
        results += model.predict(images[count:count+batch_size], conf=0.01)
        count += batch_size

    # print(results[1])
    # print(results[1].boxes)

    # print(results[1].summary(normalize=True))

    results_formatted = []

    for result in results:
        # print(f"DEBUG: result.path = {result.path}")
        file = result.path.split("/")[-1]
        confidences = result.boxes.conf.tolist()
        boxes = result.boxes.xyxyn.tolist()

        results_formatted.append([file, confidences, boxes])

    # print(results_formatted[1])

    # calculate ious
    for i, result in enumerate(results_formatted):
        ious = [0] * len(result[1])
        # Use the original image path to get the correct filename
        image_filename = os.path.basename(images[i])
        label_file = image_filename.replace(".jpg", ".txt")
        labels = read_bounding_boxes(label_directory + label_file)
        if len(labels) > 0:
            for label in labels:
                for i in range(len(result[1])):
                    new_iou = calculate_iou(label, result[2][i])
                    if new_iou > ious[i]:
                        ious[i] = new_iou
        result.append(ious)

    # print(results_formatted)

    true_labels = []
    confidence_values = []
    for result in results_formatted:
        for i in range(len(result[1])):
            confidence_values.append(result[1][i])
            true_labels.append(1 if result[3][i] >= iou_threshold else 0)

    return true_labels, confidence_values


def plot_roc_curve(ax, true_labels, confidence_values, name=None):
    display = RocCurveDisplay.from_predictions(true_labels, confidence_values, ax=ax, name=name)
    # display.plot(ax=ax, name=name)


def plot_roc_curve_detailed(fig, true_labels, confidence_values, name=None):
    """
    Fügt einem Plotly-Figure-Objekt eine ROC-Kurve hinzu, bei der an jedem Punkt der Schwellenwert angezeigt wird.

    Parameter:
      - fig: Ein existierendes Plotly-Figure-Objekt, dem die ROC-Kurve hinzugefügt wird.
      - true_labels: Liste der wahren Label.
      - confidence_values: Liste der Konfidenzwerte.
      - name: Name des Modells (wird in der Legende angezeigt).
    """

    # ROC-Berechnung
    fpr, tpr, thresholds = roc_curve(true_labels, confidence_values)

    # Trace für die ROC-Kurve hinzufügen: Linien + Marker, mit jedem Threshold als text
    text_labels = [f"Threshold: {thr:.3f}" for thr in thresholds]

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines+markers',
        name=f"ROC {name}",
        text=text_labels,
        hovertemplate="FPR: %{x:.2f}<br>TPR: %{y:.2f}<br>%{text}<extra></extra>",
        visible=True  # Die Sichtbarkeit kann über die Legende geändert werden
    ))

    # Diagonale Referenzlinie (Random Classifier) nur einmal hinzufügen, falls noch nicht vorhanden
    if not any(trace.name == "Random" for trace in fig.data):
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random',
            hoverinfo='skip'
        ))

    fig.update_layout(
        title="ROC-curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend_title="models"
    )


def check_images(images):
    images_new = []
    for i in images:
        try:
            Image.open(i).load()
            images_new.append(i)
        except Exception as e:
            # print(e)
            #  print(i)
            pass
    return images_new


def calculate_iou(box1, box2):
    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate the area of the intersection rectangle
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate the area of the union
    union_area = box1_area + box2_area - intersection_area

    # Calculate the IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def read_bounding_boxes(file_path):
    bounding_boxes = []

    with open(file_path, 'r') as file:
        for line in file:
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



