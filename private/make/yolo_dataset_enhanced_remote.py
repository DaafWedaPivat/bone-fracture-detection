import os
import shutil
import random
import cv2
import numpy as np

def process_image(src_path, dst_path):
    """Apply CLAHE to the image and save it to dst_path."""
    img = cv2.imread(src_path)
    if img is None:
        return
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    cv2.imwrite(dst_path, cl1)

def main():
    random.seed(1)
    
    # Paths (relative to the script location or passed as env)
    root = os.getenv("ROOT_DIR", "./")
    dependencies = os.path.join(root, "private/dependencies/")
    generated = os.path.join(root, "private/generated/")
    dataset = os.path.join(dependencies, "FracAtlas/")
    dataset_images = os.path.join(dataset, "images/")
    dataset_annotations = os.path.join(dataset, "Annotations/YOLO/")
    yolo_dataset = os.path.join(generated, "yolo_dataset_enhanced/")

    print(f"Preparing dataset at: {yolo_dataset}")

    # Prepare directories
    shutil.rmtree(yolo_dataset, ignore_errors=True)
    for subset in ["train", "valid", "test"]:
        os.makedirs(os.path.join(yolo_dataset, subset, "images"), exist_ok=True)
        os.makedirs(os.path.join(yolo_dataset, subset, "labels"), exist_ok=True)

    # Create dataset.yaml
    yaml_content = f"""
train: {os.path.abspath(os.path.join(yolo_dataset, "train/images"))}
val: {os.path.abspath(os.path.join(yolo_dataset, "valid/images"))}
test: {os.path.abspath(os.path.join(yolo_dataset, "test/images"))}

nc: 1
names: ["fractured"]"""
    
    with open(os.path.join(yolo_dataset, "dataset.yaml"), "w") as f:
        f.write(yaml_content)

    # List images
    fractured_images = os.listdir(os.path.join(dataset_images, "Fractured/"))
    non_fractured_images = os.listdir(os.path.join(dataset_images, "Non_fractured/"))

    # Split dataset
    test_portion = 0.05
    valid_portion = 0.25

    def split(images):
        random.shuffle(images)
        n = len(images)
        test_n = int(n * test_portion)
        valid_n = int(n * valid_portion)
        return images[:test_n], images[test_n:test_n+valid_n], images[test_n+valid_n:]

    f_test, f_valid, f_train = split(fractured_images)
    nf_test, nf_valid, nf_train = split(non_fractured_images)

    def process_subset(images, source_subdir, target_subset):
        for image in images:
            src = os.path.join(dataset_images, source_subdir, image)
            dst = os.path.join(yolo_dataset, target_subset, "images", image)
            process_image(src, dst)
            
            label = image.replace(".jpg", ".txt")
            label_src = os.path.join(dataset_annotations, label)
            label_dst = os.path.join(yolo_dataset, target_subset, "labels", label)
            if os.path.exists(label_src):
                shutil.copyfile(label_src, label_dst)

    print("Processing train...")
    process_subset(f_train, "Fractured", "train")
    process_subset(nf_train, "Non_fractured", "train")
    print("Processing validation...")
    process_subset(f_valid, "Fractured", "valid")
    process_subset(nf_valid, "Non_fractured", "valid")
    print("Processing test...")
    process_subset(f_test, "Fractured", "test")
    process_subset(nf_test, "Non_fractured", "test")

    print("Dataset preparation complete.")

if __name__ == "__main__":
    main()
