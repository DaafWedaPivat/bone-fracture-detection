from ultralytics import YOLO
import os

def main():
    # Use environment variables for flexibility
    name = os.getenv("TRAIN_NAME", "yolo11n10_enhanced")
    dataset_path = os.getenv("DATASET_PATH", "private/generated/yolo_dataset_enhanced/dataset.yaml")
    base_model = os.getenv("BASE_MODEL", "yolo11n.pt")
    epochs = int(os.getenv("EPOCHS", "10"))
    imgsz = int(os.getenv("IMGSZ", "640"))

    print(f"Starting training: {name}")
    print(f"Dataset: {dataset_path}")
    print(f"Base Model: {base_model}")

    # Load model
    model = YOLO(base_model)

    # Train
    train_results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=imgsz,
        name=name,
    )

    print("Training finished.")

if __name__ == "__main__":
    main()
