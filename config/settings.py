import os
import sys
import glob

embed_model_name = "intfloat/multilingual-e5-large-instruct"
rerank_model_name = 'netease-youdao/bce-reranker-base_v1'

""" 1. 核心路徑基礎設定 (自動識別開發或打包環境) """
if getattr(sys, 'frozen', False):
    # PyInstaller 打包環境：sys._MEIPASS 即為暫存根目錄
    BASE_PATH = sys._MEIPASS
else:
    # 開發環境：因為 settings.py 在 config/ 資料夾內，所以向上推兩層回到專案根目錄
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


""" 2. 定義所有應用程式目錄 """
CHROMADB_DIR = os.path.join(BASE_PATH, "chromadb_storage")
RAW_DATA_DIR = os.path.join(BASE_PATH, "data", "raw")
OUTPUT_TABLES_DIR = os.path.join(BASE_PATH, "data", "processed", "out_tables")

MODEL_WEIGHTS_ROOT = os.path.join(BASE_PATH, "model_weights")
EMBED_WEIGHTS_DIR = os.path.join(MODEL_WEIGHTS_ROOT, "embed")
RERANK_WEIGHTS_DIR = os.path.join(MODEL_WEIGHTS_ROOT, "rerank")

TESSERACT_DIR = os.path.join(BASE_PATH, "Tesseract-OCR")
TESSDATA_DIR = os.path.join(BASE_PATH, "Tesseract")


""" 3. 一鍵建立所有必要資料夾 """
NEED_DIRS = [CHROMADB_DIR, RAW_DATA_DIR, OUTPUT_TABLES_DIR, MODEL_WEIGHTS_ROOT]
for dir_path in NEED_DIRS:
    os.makedirs(dir_path, exist_ok=True)


""" 4. 模型權重與離線模式檢查 """
def check_hf_cached_model(cache_dir):
    if not os.path.exists(cache_dir):
        return False
    # 檢查 Hugging Face 結構
    if glob.glob(os.path.join(cache_dir, "models--*")):
        return True
    # 檢查直接放權重的情況
    return os.path.exists(os.path.join(cache_dir, "config.json"))

embed_ready = check_hf_cached_model(EMBED_WEIGHTS_DIR)
rerank_ready = check_hf_cached_model(RERANK_WEIGHTS_DIR)

if embed_ready and rerank_ready:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("[System Cache] 本地權重偵測成功！已全面鎖定離線模式（封鎖 HF 網路請求）。")
else:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    print("[System Cache] 偵測到缺失權重。暫時允許網路連線以利下載。")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


""" 5. Tesseract OCR 環境變數設定 """
if os.path.exists(TESSERACT_DIR):
    os.environ["PATH"] += os.pathsep + TESSERACT_DIR

if os.path.exists(TESSDATA_DIR):
    # 修正：這裡應該帶入 TESSDATA_DIR 變數，而不是字串 "TESSDATA_PREFIX"
    os.environ["TESSDATA_PREFIX"] = TESSDATA_DIR