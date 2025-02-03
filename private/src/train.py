import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import ultralytics
    ultralytics.checks()
    return (ultralytics,)


@app.cell
def _():
    from ultralytics import YOLO

    dataset = "../bone-fracture-detection/private/generated/yolo_dataset/dataset.yaml"

    _ = YOLO("private/generated/yolo11n.pt")
    model = YOLO("private/generated/yolov8s.pt")

    train_results = model.train(
        data=dataset,  # path to dataset YAML
        epochs=30,  # number of training epochs
        imgsz=640,  # training image size
        project="private/generated/"
    )
    return YOLO, dataset, model, train_results


if __name__ == "__main__":
    app.run()
