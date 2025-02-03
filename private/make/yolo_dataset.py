import marimo

__generated_with = "0.10.19"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import shutil
    import glob
    import random

    random.seed(1)
    return glob, mo, os, random, shutil


@app.cell
def _():
    root = ""
    dependencies = root + "private/dependencies/"
    generated = root + "private/generated/"
    dataset = dependencies + "FracAtlas/"
    dataset_images = dataset + "images/"
    dataset_annotations = dataset + "Annotations/YOLO/"
    yolo_dataset = generated + "yolo_dataset/"
    return (
        dataset,
        dataset_annotations,
        dataset_images,
        dependencies,
        generated,
        root,
        yolo_dataset,
    )


@app.cell
def _(dataset, os):
    fractured_images = os.listdir(dataset + "images/Fractured/")
    non_fractured_images = os.listdir(dataset + "images/Non_fractured/")

    print("fractured images: ", len(fractured_images))
    print("non fractured images: ", len(non_fractured_images))
    return fractured_images, non_fractured_images


@app.cell
def _(mo):
    mo.md(r"""## Prepare dataset folder""")
    return


@app.cell
def _(os, shutil, yolo_dataset):
    shutil.rmtree(yolo_dataset)

    os.makedirs(yolo_dataset + "train/images", exist_ok=True)
    os.makedirs(yolo_dataset + "train/labels", exist_ok=True)
    os.makedirs(yolo_dataset + "valid/images", exist_ok=True)
    os.makedirs(yolo_dataset + "valid/labels", exist_ok=True)
    os.makedirs(yolo_dataset + "test/images", exist_ok=True)
    os.makedirs(yolo_dataset + "test/labels", exist_ok=True)

    file_content = """\
    train: ../train/images
    val: ../valid/images
    test: ../test/images

    nc: 1
    names: ["fractured"]"""

    with open(yolo_dataset + "dataset.yaml", "w") as f:
        f.write(file_content)

    return f, file_content


@app.cell
def _(mo):
    mo.md(r"""## Split Dataset""")
    return


@app.cell
def _(fractured_images, non_fractured_images, random):
    train_portion = 0.7
    valid_portion = 0.25
    test_portion = 0.05

    num_fractured_test_images = int(len(fractured_images) * test_portion)
    num_fractured_valid_images = int(len(fractured_images) * valid_portion)

    num_non_fractured_test_images = int(len(non_fractured_images) * test_portion)
    num_non_fractured_valid_images = int(len(non_fractured_images) * valid_portion)

    fractured_test_images = random.sample(fractured_images, num_fractured_test_images)
    for e in fractured_test_images:
        fractured_images.remove(e)

    non_fractured_test_images = random.sample(non_fractured_images, num_non_fractured_test_images)
    for e in non_fractured_test_images:
        non_fractured_images.remove(e)

    fractured_valid_images = random.sample(fractured_images, num_fractured_valid_images)
    for e in fractured_valid_images:
        fractured_images.remove(e)

    non_fractured_valid_images = random.sample(non_fractured_images, num_non_fractured_valid_images)
    for e in non_fractured_valid_images:
        non_fractured_images.remove(e)

    fractured_train_images = fractured_images
    non_fractured_train_images = non_fractured_images

    print(f"train images fractured / non fractured: {len(fractured_train_images)} / {len(non_fractured_train_images)}")
    print(f"validation images fractured / non fractured: {len(fractured_valid_images)} / {len(non_fractured_valid_images)}")
    print(f"test images fractured / non fractured: {len(fractured_test_images)} / {len(non_fractured_test_images)}")
    return (
        e,
        fractured_test_images,
        fractured_train_images,
        fractured_valid_images,
        non_fractured_test_images,
        non_fractured_train_images,
        non_fractured_valid_images,
        num_fractured_test_images,
        num_fractured_valid_images,
        num_non_fractured_test_images,
        num_non_fractured_valid_images,
        test_portion,
        train_portion,
        valid_portion,
    )


@app.cell
def _(mo):
    mo.md(r"""## Copy files""")
    return


@app.cell
def _(
    dataset_annotations,
    dataset_images,
    fractured_test_images,
    fractured_train_images,
    fractured_valid_images,
    non_fractured_test_images,
    non_fractured_train_images,
    non_fractured_valid_images,
    shutil,
    yolo_dataset,
):
    train_images = fractured_train_images + non_fractured_train_images
    valid_images = fractured_valid_images + non_fractured_valid_images
    test_images = fractured_test_images + non_fractured_test_images


    def get_annotation_file(image_file: str) -> str:
        return image_file.replace(".jpg", ".txt")


    for image in fractured_train_images:
        shutil.copyfile(dataset_images + "Fractured/" + image, yolo_dataset + "train/images/" + image)
        label = get_annotation_file(image)
        shutil.copyfile(dataset_annotations + label, yolo_dataset + "train/labels/" + label)

    for image in non_fractured_train_images:
        shutil.copyfile(dataset_images + "Non_fractured/" + image, yolo_dataset + "train/images/" + image)
        label = get_annotation_file(image)
        shutil.copyfile(dataset_annotations + label, yolo_dataset + "train/labels/" + label)

    for image in fractured_valid_images:
        shutil.copyfile(dataset_images + "Fractured/" + image, yolo_dataset + "valid/images/" + image)
        label = get_annotation_file(image)
        shutil.copyfile(dataset_annotations + label, yolo_dataset + "valid/labels/" + label)

    for image in non_fractured_valid_images:
        shutil.copyfile(dataset_images + "Non_fractured/" + image, yolo_dataset + "valid/images/" + image)
        label = get_annotation_file(image)
        shutil.copyfile(dataset_annotations + label, yolo_dataset + "valid/labels/" + label)

    for image in fractured_test_images:
        shutil.copyfile(dataset_images + "Fractured/" + image, yolo_dataset + "test/images/" + image)
        label = get_annotation_file(image)
        shutil.copyfile(dataset_annotations + label, yolo_dataset + "test/labels/" + label)

    for image in non_fractured_test_images:
        shutil.copyfile(dataset_images + "Non_fractured/" + image, yolo_dataset + "test/images/" + image)
        label = get_annotation_file(image)
        shutil.copyfile(dataset_annotations + label, yolo_dataset + "test/labels/" + label)

    return (
        get_annotation_file,
        image,
        label,
        test_images,
        train_images,
        valid_images,
    )


if __name__ == "__main__":
    app.run()
