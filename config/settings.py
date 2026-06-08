import os
import sys
import glob

"""" 設定chromadb """

# 💡 支援 pyinstaller 打包(但傾向不打包，整檔太大)
if getattr(sys, 'frozen', False):
    # 在 PyInstaller exe 环境中，sys._MEIPASS 指向临时解压目录
    BASE_PATH = sys._MEIPASS
else:
    # 开发环境：基于 config.py 所在位置推导
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMADB_DIR = os.path.join(BASE_PATH, "chromadb_storage")

# 确保目录存在
os.makedirs(CHROMADB_DIR, exist_ok=True)

""" 設定模型權重路徑和 Hugging Face 緩存行為 """

# 💡 支持 PyInstaller 打包环境
if getattr(sys, 'frozen', False):
    # 在 PyInstaller exe 环境中，sys._MEIPASS 指向临时解压目录
    BASE_PATH = sys._MEIPASS
else:
    # 开发环境：基于 config.py 所在位置推导
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 💡 Define absolute paths for local dedicated model weights storage
MODEL_WEIGHTS_ROOT = os.path.join(BASE_PATH, "model_weights")
EMBED_WEIGHTS_DIR = os.path.join(MODEL_WEIGHTS_ROOT, "bge-m3")
RERANK_WEIGHTS_DIR = os.path.join(MODEL_WEIGHTS_ROOT, "bge-reranker")

# 💡 Data directories
RAW_DATA_DIR = os.path.join(BASE_PATH, "data", "raw")
CHROMADB_DIR = os.path.join(BASE_PATH, "chromadb_storage")

# 💡 Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, CHROMADB_DIR, MODEL_WEIGHTS_ROOT]:
    os.makedirs(dir_path, exist_ok=True)

# 💡 Check if models have already been downloaded locally
# Check for Hugging Face cache structure (models--BAAI--bge-m3/snapshots/...)
def check_hf_cached_model(cache_dir):
    if not os.path.exists(cache_dir):
        return False
    # Check for Hugging Face cache folder structure
    hf_cache_pattern = os.path.join(cache_dir, "models--*")
    if glob.glob(hf_cache_pattern):
        return True
    # Fallback: check for direct config.json
    return os.path.exists(os.path.join(cache_dir, "config.json"))

# 💡 Check both models that are used in the application
embed_ready = check_hf_cached_model(EMBED_WEIGHTS_DIR)
rerank_ready = check_hf_cached_model(RERANK_WEIGHTS_DIR)

if embed_ready and rerank_ready:
    # 💡 Fully lock into offline mode if local weights exist to boost speed to milliseconds
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("[System Cache] Local weights detected. Hugging Face network requests are now fully BLOCKED.")
else:
    # 💡 Allow network connection if any model file is missing or manually deleted
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    print("[System Cache] Missing weights or cleared by user. Internet connection is TEMPORARILY ALLOWED for download.")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Tesseract variables initialization
TESSERACT_DIR = os.path.join(BASE_PATH, "Tesseract-OCR")
TESSDATA_DIR = os.path.join(BASE_PATH, "Tesseract")

if os.path.exists(TESSERACT_DIR):
    os.environ["PATH"] += os.pathsep + TESSERACT_DIR
if os.path.exists(TESSDATA_DIR):
    os.environ["TESSDATA_PREFIX"] = "TESSDATA_PREFIX"

CHROMADB_DIR = os.path.join(BASE_PATH, "chromadb_storage")
RAW_DATA_DIR = os.path.join(BASE_PATH, "data", "raw")
OUTPUT_TABLES_DIR = os.path.join(BASE_PATH, "data", "processed", "out_tables")

os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_TABLES_DIR, exist_ok=True)
os.makedirs(MODEL_WEIGHTS_ROOT, exist_ok=True)