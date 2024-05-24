import nltk

# List of resources to download
resources = ['wordnet', 'punkt', 'stopwords']

# Download the necessary NLTK resources
for resource in resources:
    nltk.download(resource)

