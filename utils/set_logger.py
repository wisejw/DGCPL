import os
from datetime import datetime
import logging


def set_logger(dataset):
    if not os.path.exists(f"logs/{dataset}/"):
        os.makedirs(f"logs/{dataset}/")

    if not os.path.exists(f"best_model/{dataset}/"):
        os.makedirs(f"best_model/{dataset}/")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join("logs", f"{dataset}/training_{current_time}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

