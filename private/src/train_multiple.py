import logging

from ultralytics import YOLO
import yaml

logging.basicConfig(
    level=logging.INFO,
    filename="../generated/train.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
)


dataset = "../generated/yolo_dataset/dataset.yaml"
standard_settings = {
    "data": dataset,
    "project": "../generated/",
    "imgsz": 640,
 }

def main():
    with open("config.yaml", 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    logging.info(config)

    for setting in config:
        logging.info(f"{setting}")
        try:
            model = YOLO(f"../generated/{setting["model"]}")
            setting.pop("model")

            results = model.train(**standard_settings, **setting)

        except Exception as e:
            logging.error(e)


if __name__ == "__main__":
    main()
