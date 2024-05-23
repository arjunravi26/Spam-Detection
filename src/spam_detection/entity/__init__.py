from dataclasses import dataclass
from pathlib import Path
@dataclass
class DataIngestionConfig:
    root_dir:Path
    data_path:str
    train_path:str
    test_path:str
    validation_path:str
    
@dataclass
class DataTransformationConfig:
    root_dir:Path
    train_path:str
    test_path:str
    validation_path:str