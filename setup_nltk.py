import nltk
import gensim
from gensim.utils import simple_preprocess
from gensim.models import Word2Vec
import gensim.downloader as api
from src.spam_detection.utils.common import save_model
from src.spam_detection.config.configuration import ConfiguaraionManager

# List of resources to download
resources = ['wordnet', 'punkt', 'stopwords']

# Download the necessary NLTK resources
for resource in resources:
    nltk.download(resource)
    
word2vec_model = api.load('word2vec-google-news-300')
config = ConfiguaraionManager()
data_path = config.data_transformation()
save_model(word2vec_model,data_path.model_path)



