import os
from ensure import ensure_annotations
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from src.spam_detection.logging import logging


@ensure_annotations
def read_yaml(file_path):
    try:
        content = yaml.safe_load(file_path)
        return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e


@ensure_annotations
def create_directory(path_list: list, verbose=True):
    for path in path_list:
        os.makedirs(path, exist_ok=True)
    if verbose:
        logging.info(f"created directory at: {path}")


@ensure_annotations
def get_size(file_path) -> str:
    size_in_kb = round(os.path.getsize(filename=file_path) / 1024)
    return f"file size is {size_in_kb}"
