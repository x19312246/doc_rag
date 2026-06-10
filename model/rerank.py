from sentence_transformers import CrossEncoder
from config.settings import RERANK_WEIGHTS_DIR
from config.settings import rerank_model_name

# 💡 在模組層級初始化單例
_reranker_instance = None

def get_reranker():
    global _reranker_instance
    if _reranker_instance is None:
        # 只有在第一次呼叫時才初始化
        _reranker_instance = CrossEncoder(rerank_model_name, cache_folder=RERANK_WEIGHTS_DIR)
    return _reranker_instance