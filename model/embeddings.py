import os
from sentence_transformers import SentenceTransformer
from indexer.config import EMBED_WEIGHTS_DIR

class ChromaEmbeddingFunction:
    def __init__(self):
        # 💡 Automatically loads from or saves into the consolidated project folder
        self.model = SentenceTransformer("BAAI/bge-m3", cache_folder=EMBED_WEIGHTS_DIR)

    def __call__(self, input):
        return self.model.encode(input).tolist()
        
    def embed_query(self, input):  
        return self.__call__(input)

    def name(self):
        return "BGE-M3"