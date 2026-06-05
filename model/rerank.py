import os
from sentence_transformers import CrossEncoder
from indexer.config import RERANK_WEIGHTS_DIR

def get_reranker():
    # 💡 Automatically loads from or saves into the consolidated project folder
    return CrossEncoder('BAAI/bge-reranker-base', cache_folder=RERANK_WEIGHTS_DIR)