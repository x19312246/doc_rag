# import os
from sentence_transformers import CrossEncoder
from config.settings import RERANK_WEIGHTS_DIR
from config.settings import rerank_model_name

#model_name = 'BAAI/bge-reranker-base'
#model_name = 'netease-youdao/bce-reranker-base_v1'

def get_reranker():
    # 💡 Automatically loads from or saves into the consolidated project folder
    return CrossEncoder(rerank_model_name, cache_folder=RERANK_WEIGHTS_DIR)