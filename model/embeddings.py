#import os
from sentence_transformers import SentenceTransformer
from config.settings import EMBED_WEIGHTS_DIR
from config.settings import embed_model_name

class ChromaEmbeddingFunction:
    def __init__(self):
        # 💡 Automatically loads from or saves into the consolidated project folder
        self.model = SentenceTransformer(embed_model_name, cache_folder=EMBED_WEIGHTS_DIR)

    def __call__(self, input):
        return self.model.encode(input).tolist()
        
    def embed_query(self, input): 
        instruct_query = f"Instruct: Given a web search query, retrieve relevant passages that answer the query\n{input}" 
        #return self.__call__(input)
        return self.model.encode(instruct_query).tolist()

    def name(self):
        return embed_model_name