import os
from datetime import datetime
import logging


def set_logger(dataset):
    # Create a directory for logs if it does not exist
    if not os.path.exists(f"logs/{dataset}/"):
        os.makedirs(f"logs/{dataset}/")

    # Create a directory for saving the best model if it does not exist
    if not os.path.exists(f"best_model/{dataset}/"):
        os.makedirs(f"best_model/{dataset}/")

    # Get the current timestamp in the format 'YYYY-MM-DD_HH-MM-SS'
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join("logs", f"{dataset}/training_{current_time}.log")    # Set the log file path

    # Set up the logger configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        # Define log handlers to output both to file and console
        handlers=[
            logging.FileHandler(log_file_path, mode="w"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

