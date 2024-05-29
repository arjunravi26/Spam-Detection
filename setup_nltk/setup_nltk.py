import nltk
import gensim.downloader as api
from src.spam_detection.config.configuration import ConfiguaraionManager
from tqdm import tqdm
import gensim.downloader as api

# List of resources to download
resources = ['wordnet', 'punkt', 'stopwords']

# Download the necessary NLTK resources
for resource in tqdm(resources, desc='Downloading NLTK resources'):
    nltk.download(resource)
    
print("Loading Word2Vec model...")
# word2vec_model = api.load('word2vec-google-news-300')
word2vec_model = api.load("word2vec-google-news-300")

print("Word2Vec model loaded.")

config = ConfiguaraionManager()
data_path = config.data_transformation()

print("Saving model...")
word2vec_model.save(data_path.model_path)
print("Model saved.")
