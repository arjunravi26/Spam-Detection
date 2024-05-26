import os
from ensure import ensure_annotations
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from src.spam_detection.logging import logging
from pathlib import Path
import pickle
import pandas as pd
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (Path): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            if not content:
                logging.error(f"YAML file: {path_to_yaml} is empty or not valid.")
                raise ValueError("YAML file is empty or not valid.")
            logging.info(f"YAML file: {path_to_yaml} loaded successfully with content: {content}")
            return ConfigBox(content)
    except BoxValueError as e:
        logging.error(f"Error loading ConfigBox: {e}")
        raise ValueError(f"YAML file is empty or not valid.{path_to_yaml}")
    except Exception as e:
        logging.error(f"Unexpected error reading YAML file: {e}")
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
@ensure_annotations

def save_data(path,X,y):
    df = pd.concat([pd.DataFrame(x.reshape(1, -1)) for x in X], ignore_index=True)
    df['output'] = y
    df.to_csv(path,index=False)
    logging.info(f"Transformed data saved to {path}")
    
@ensure_annotations
def save_model(model, path):
    with open(path, 'wb') as file:
        pickle.dump(model, file)
    logging.info(f"Model saved to {path}")

