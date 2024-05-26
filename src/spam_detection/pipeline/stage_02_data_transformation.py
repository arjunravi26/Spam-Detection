from src.spam_detection.config.configuration import ConfiguaraionManager
from src.spam_detection.component.data_transformation import DataTransformation
class DataTransformationPipeline:
    def __init__(self,model) -> None:
        self.model = model
    def start(self):
        config = ConfiguaraionManager()
        data_transformation_config = config.data_transformation()
        data_transformation = DataTransformation(config=data_transformation_config,model=self.model)
        data_transformation.preprocess_data()
    def predict_transform(self,data):
        config = ConfiguaraionManager()
        data_transformation_config = config.data_transformation()
        data_transformation = DataTransformation(config=data_transformation_config,model=self.model)
        data = data_transformation.preprocess_new_data(data=data)
        return data
        