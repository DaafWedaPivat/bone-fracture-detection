import marimo

__generated_with = "0.10.17"
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
    path = Path('C:/Users/darwin/Daten/Dokumente/Programmierung/marimo/marimo-env/FracAtlas/FracAtlas/images')
    path_Fractured = Path('C:/Users/darwin/Daten/Dokumente/Programmierung/marimo/marimo-env/FracAtlas/FracAtlas/images/Fractured')
    path_Non_fractured = Path('C:/Users/darwin/Daten/Dokumente/Programmierung/marimo/marimo-env/FracAtlas/FracAtlas/images/Non_fractured')
    return path, path_Fractured, path_Non_fractured


@app.cell
def _(Image, os, warnings):
    def verify_and_load_image(filepath):
        try:
            if not os.path.exists(filepath):
                print(f"File does not exist: {filepath}")
                return False

            file_size = os.path.getsize(filepath)
            if file_size == 0:
                print(f"File is empty: {filepath}")
                return False

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                with Image.open(filepath) as img:
                    img.load()
                    print(f"\nImage details for {filepath}:")
                    print(f"Format: {img.format}")
                    print(f"Size: {img.size}")
                    print(f"Mode: {img.mode}")
                    if img.mode != 'RGB':
                        print(f"Converting {filepath} from {img.mode} to RGB")
                        img = img.convert('RGB')
            return True
        except Exception as e:
            print(f'\nDetailed error for {filepath}:')
            print(f'Error type: {type(e).__name__}')
            print(f'Error message: {str(e)}')
            return False
    return (verify_and_load_image,)


@app.cell
def _(get_image_files, progress_bar, verify_and_load_image):
    def get_valid_images(path):
        all_files = get_image_files(path)
        print(f"\nFound {len(all_files)} total files")

        valid_files = []
        corrupted_files = []

        for file in progress_bar(all_files):
            if verify_and_load_image(file):
                valid_files.append(file)
            else:
                corrupted_files.append(file)
        print('\n=== Verification Summary ===')
        print(f'Total files found: {len(all_files)}')
        print(f'Valid files: {len(valid_files)}')
        print(f'Corrupted files: {len(corrupted_files)}')

        if corrupted_files:
            print('\nProblematic files:')
            for f in corrupted_files:
                print(f'- {f}')
        return valid_files
    return (get_valid_images,)


app._unparsable_cell(
    r"""
    def label_fracture_files(file_path):
        file_path = str(file_path)
        base_path = str(path)

        if 'Fractured' in file_path:
            return 'fractured'
        else: 'Non_fractured' in file_path:
            return 'non_fractured'
    """,
    name="_"
)


@app.cell
def _(
    CategoryBlock,
    DataBlock,
    ImageBlock,
    IntToFloatTensor,
    RandomSplitter,
    Resize,
    get_valid_images,
    label_fracture_files,
    path,
):
    valid_files = get_valid_images(path)

    dls = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=lambda _: valid_files,  # Fixed lambda syntax
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=label_fracture_files,
            item_tfms=[Resize(192, method='squish')],
            batch_tfms=[IntToFloatTensor]
    ).dataloaders(path, bs=16, num_workers=0)

    print(f'\nDataset sizes:')
    print(f'Training: {len(dls.train_ds)}')
    print(f'Validation: {len(dls.valid_ds)}')

    print('\nTesting batch loading...')
    xb, yb = dls.one_batch()
    print('Successfully loaded a batch!')
    print(f'Batch shape: {xb.shape}')
    print(f'Labels shape: {yb.shape}')
    dls.show_batch()
    return dls, valid_files, xb, yb


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
