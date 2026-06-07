import os
from sentence_transformers import CrossEncoder
from indexer.config import RERANK_WEIGHTS_DIR

#model_name = 'BAAI/bge-reranker-base'
#model_name = 'netease-youdao/bce-reranker-base_v1'

def get_reranker():
    # 💡 Automatically loads from or saves into the consolidated project folder
    return CrossEncoder('netease-youdao/bce-reranker-base_v1', cache_folder=RERANK_WEIGHTS_DIR)