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

    name = "yolo11n"

    dataset = "../generated/yolo_dataset/dataset.yaml"

    # _ = YOLO("private/generated/yolo11n.pt")
    model = YOLO("../generated/yolo11n.pt")

    train_results = model.train(
        data=dataset,
        project="../generated/",
        epochs=100,
        imgsz=640,
        name=name,
    )
    return YOLO, dataset, model, train_results


if __name__ == "__main__":
    app.run()
