import marimo

__generated_with = "0.10.19"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np 
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import os
    import warnings
    import fastai.vision.all
    from fastai.vision.all import (
        get_image_files,
        ImageBlock,
        CategoryBlock,
        DataBlock,
        RandomSplitter,
        PILImage, 
        Resize,
        progress_bar,
        IntToFloatTensor,
        vision_learner,
        resnet34,
        error_rate
    )

    from pathlib import Path
    from PIL import Image, ImageFile
    return (
        CategoryBlock,
        DataBlock,
        Image,
        ImageBlock,
        ImageFile,
        IntToFloatTensor,
        PILImage,
        Path,
        RandomSplitter,
        Resize,
        error_rate,
        fastai,
        get_image_files,
        np,
        os,
        pd,
        progress_bar,
        resnet34,
        vision_learner,
        warnings,
    )


@app.cell
def _(ImageFile, warnings):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    warnings.filterwarnings('ignore', category=UserWarning)
    return


@app.cell
def _(Path):
    path = Path('private/dependencies/FracAtlas/images')
    path_Fractured = Path('private/dependencies/FracAtlas/images/Fractured')
    path_Non_fractured = Path('private/dependencies/FracAtlas/images/Non_fractured')
    return path, path_Fractured, path_Non_fractured


@app.cell
def _():
    return


@app.cell
def _(path):
    def label_fracture_files(file_path):
        file_path = str(file_path)
        base_path = str(path)

        if 'Fractured' in file_path:
            return 'fractured'
        elif 'Non_fractured' in file_path:
            return 'non_fractured'
    return (label_fracture_files,)


@app.cell
def _(
    CategoryBlock,
    DataBlock,
    ImageBlock,
    IntToFloatTensor,
    RandomSplitter,
    Resize,
    get_image_files,
    label_fracture_files,
    path,
):
    all_files = get_image_files(path)


    dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=lambda _: all_files,  # Fixed lambda syntax
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=label_fracture_files,
            item_tfms=[Resize(192, method='squish')],
            batch_tfms=[IntToFloatTensor]
    ).dataloaders(path, bs=16, num_workers=0)

    print(f'\nDataset sizes:')
    print(f'Training: {len(dls.train_ds)}')
    print(f'Validation: {len(dls.valid_ds)}')

    dls.show_batch()
    return all_files, dls


@app.cell
def _(dls, error_rate, resnet34, vision_learner):
    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(1)
    return (learn,)


@app.cell
def _(Image, learn, path):
    im1 = Image.open(str(path) + "/Fractured/IMG0000019.jpg")
    im1.to_thumb(256,256)
    learn.predict(im1)
    return (im1,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
