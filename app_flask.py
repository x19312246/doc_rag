import os
import sys
import hashlib
import shutil
import threading
import logging
import time
from logging.handlers import TimedRotatingFileHandler
from flask import Flask, render_template, request, jsonify
import json

import config.settings  

if getattr(sys, 'frozen', False):
    project_root = sys._MEIPASS
else:
    project_root = os.path.dirname(os.path.abspath(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from indexer.ocr_loader import extract_pdf_pages_info, convert_pages_to_chunks, reconstruct_pages_via_vlm
from config.settings import RAW_DATA_DIR, CHROMADB_DIR
from indexer.indexer import build_vector_index
from retriever.retriever import execute_rag_retrieval
from model.llm import query_llm, get_local_models

# 從 config.settings 引入 CHROMADB_DIR
from config.settings import RAW_DATA_DIR, CHROMADB_DIR

# 將 NOTES_FILE 的路徑直接指向 CHROMADB_DIR 內
NOTES_FILE = os.path.join(CHROMADB_DIR, "database_notes.json")

# =====================================================================
# FLASK APPLICATION SETUP & ENVIRONMENT RELOADER CONTROL
# =====================================================================

app = Flask(__name__, template_folder="templates")

# Initialize Thread Lock
status_lock = threading.Lock()

# Global Task Status initialization controlled via Werkzeug Environment variable
if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    print("[Flask Reloader] Real child process detected. Initializing TASK_STATUS cluster.")
    TASK_STATUS = {
        "ocr": {"running": False, "msg": "Idle", "success": True},
        "vlm": {"running": False, "msg": "Idle", "success": True},
        "query": {"running": False, "msg": "Idle", "success": True}
    }
else:
    print("[Flask Reloader] Parent management process tracking standard boot sequence.")
    TASK_STATUS = {
        "ocr": {"running": False, "msg": "Manager Core Initializing", "success": True},
        "vlm": {"running": False, "msg": "Manager Core Initializing", "success": True},
        "query": {"running": False, "msg": "Manager Core Initializing", "success": True}
    }

def preload_and_verify_weights():
    """ 
    Proactively checks and preloads embedding & rerank models on startup.
    Object references are deleted immediately after verification to ensure runtime safety.
    """
    print("\n==================================================")
    print("[Model Weight Initialization] Verifying local Embedding & Reranker configurations...")
    try:
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from config.settings import embed_model_name, rerank_model_name, EMBED_WEIGHTS_DIR, RERANK_WEIGHTS_DIR
        
        print(f" -> Inspecting Embedding target: {embed_model_name}")
        embed_model = SentenceTransformer(embed_model_name, cache_folder=EMBED_WEIGHTS_DIR)
        del embed_model
        
        print(f" -> Inspecting Reranker target: {rerank_model_name}")
        rerank_model = CrossEncoder(rerank_model_name, cache_folder=RERANK_WEIGHTS_DIR)
        del rerank_model
        
        print("[Model Weight Initialization] Status: All model configurations verified and ready.")
        embed_ok = True
        rerank_ok = True
    except Exception as e:
        print(f"[CRITICAL] Model weight preloading workflow encountered a failure: {e}")
        embed_ok = False
        rerank_ok = False

    if embed_ok and rerank_ok:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("[Model Weight Initialization] Offline lock engaged: HF network requests are now fully blocked.")
    else:
        print("[Model Weight Initialization] WARNING: One or more models failed — offline lock NOT engaged.")
    print("==================================================\n")

# Run preloading only in the active worker process to prevent double memory usage
if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    preload_and_verify_weights()

# =====================================================================
# LOGGING CONFIGURATION & NOISE REDUCTION
# =====================================================================

LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

file_handler = TimedRotatingFileHandler(
    os.path.join(LOGS_DIR, "rag_system_daily.log"),
    when="D",
    interval=1,
    backupCount=30,
    encoding="utf-8"
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

app_logger = logging.getLogger('app_status_logger')
app_logger.setLevel(logging.INFO)
app_logger.addHandler(file_handler)

werkzeug_log = logging.getLogger('werkzeug')
for h in werkzeug_log.handlers[:]:
    werkzeug_log.removeHandler(h)

class PollingNoiseFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        if "/api/task_status/" in msg:
            return False  
        return True

file_handler.addFilter(PollingNoiseFilter())

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter('%(message)s'))
console_handler.addFilter(PollingNoiseFilter())
werkzeug_log.addHandler(console_handler)
app_logger.addHandler(console_handler)

# =====================================================================
# TASK CANCELLATION ENGINE SETUP
# =====================================================================

class TaskCancellation:
    def __init__(self):
        self.is_running = True

ACTIVE_CANCELLATIONS = {
    "ocr": TaskCancellation(),
    "vlm": TaskCancellation(),
    "query": TaskCancellation()
}

# -------------------------------------------------------------------------
# BACKGROUND WORKER FUNCTIONS
# -------------------------------------------------------------------------

def background_ocr_worker(file_path, current_generated_id, start_page, end_page, *args, **kwargs):
    global TASK_STATUS, ACTIVE_CANCELLATIONS
    cancel_token = ACTIVE_CANCELLATIONS["ocr"]

    source_file_name = os.path.splitext(os.path.basename(file_path))[0]

    try:
        with status_lock:
            TASK_STATUS["ocr"]["step"] = "Executing primary PDF structural layout extraction pipeline..."

        pages_info = extract_pdf_pages_info(file_path, start_page=start_page, end_page=end_page)

        if not cancel_token.is_running:
            return

        with status_lock:
            TASK_STATUS["ocr"]["step"] = "Synchronizing physical image assets with unique identity hash scope..."

        # 建立此次任務專屬隔離圖片目錄
        vlm_target_dir = os.path.join(RAW_DATA_DIR, current_generated_id)
        os.makedirs(vlm_target_dir, exist_ok=True)

        # 複製圖片到隔離目錄並同步更新 pages_info 內的路徑
        # 這樣 convert_pages_to_chunks 寫入 metadata 的 local_img_path 才指向永久隔離目錄
        # 而非共用暫存目錄（out_images 可能被下次 OCR 覆蓋）
        for p_data in pages_info:
            master_img = p_data.get("master_page_image", "")
            if master_img and os.path.exists(master_img):
                dst_file = os.path.join(vlm_target_dir, os.path.basename(master_img))
                shutil.copy2(master_img, dst_file)
                p_data["master_page_image"] = dst_file
            for img_item in p_data.get("images", []):
                img_path = img_item.get("file_path", "")
                if img_path and os.path.exists(img_path):
                    dst_img = os.path.join(vlm_target_dir, os.path.basename(img_path))
                    shutil.copy2(img_path, dst_img)
                    img_item["file_path"] = dst_img

        if not cancel_token.is_running:
            return

        with status_lock:
            TASK_STATUS["ocr"]["step"] = "Transforming localized structural tokens into vector embedding chunks..."

        chunks = convert_pages_to_chunks(
            pages_info,
            source_name=source_file_name,
            start_page=start_page,
            end_page=end_page
        )

        if not cancel_token.is_running:
            return

        with status_lock:
            TASK_STATUS["ocr"]["step"] = "Committing base text layers into isolated ChromaDB OCR collection..."

        total_inserted = build_vector_index(chunks, f"{current_generated_id}_ocr")

        with status_lock:
            TASK_STATUS["ocr"]["running"] = False
            TASK_STATUS["ocr"]["step"] = "Completed successfully"
            TASK_STATUS["ocr"]["progress"] = 100
            TASK_STATUS["ocr"]["chunks"] = total_inserted

    except Exception as e:
        with status_lock:
            TASK_STATUS["ocr"]["running"] = False
            TASK_STATUS["ocr"]["step"] = f"Error: {str(e)}"
            TASK_STATUS["ocr"]["progress"] = 0

def background_vlm_worker(target_doc_id, provider, model_name, target_ip, target_port, cancel_token):
    global TASK_STATUS
    try:
        ocr_source_id = f"{target_doc_id}_ocr"
        new_chunks = reconstruct_pages_via_vlm(
            ocr_source_id,
            provider,
            model_name,
            target_ip,
            target_port,
            worker_thread=cancel_token
        )
        if not cancel_token.is_running:
            with status_lock:  
                TASK_STATUS["vlm"] = {"running": False, "msg": "VLM processing cancelled by user.", "success": False}
            return
        if not new_chunks:
            with status_lock:  
                TASK_STATUS["vlm"] = {"running": False, "msg": "No image source assets found or VLM response was blank.", "success": False}
            return
            
        total_inserted = build_vector_index(new_chunks, f"{target_doc_id}_vlm")
        with status_lock:  
            TASK_STATUS["vlm"] = {"running": False, "msg": f"Success! Written {total_inserted} chunk blocks.", "success": True}
    except Exception as err:
        with status_lock:  
            TASK_STATUS["vlm"] = {"running": False, "msg": str(err), "success": False}

def background_query_worker(user_query, target_id, provider, model_name, api_key, target_ip, target_port, cancel_token):
    global TASK_STATUS
    try:
        context = execute_rag_retrieval(user_query, target_id)
        
        if not cancel_token.is_running:
            with status_lock:  
                TASK_STATUS["query"] = {"running": False, "msg": "Task cancelled by user.", "success": False}
            return
            
        full_prompt = f"""你是一個專業的本地知識庫AI助手。請嚴格根據以下提供的【參考文本】來精準回答使用者的問題。
如果參考文本中找不到答案，請委婉告知「無法從目前文件中找到相關解答」，切勿編造事實。

【參考文本】：
{context}

=========================================

【使用者的問題】：
{user_query}

請提供條理清晰的繁體中文回答："""
        
        p_clean = provider.lower().strip()
        if "lmstudio" in p_clean:
            p_val = "lmstudio"
        elif "ollama" in p_clean:
            p_val = "ollama"
        else:
            p_val = "groq"
        
        answer = query_llm(
            prompt=full_prompt,
            provider=p_val,
            model_name=model_name,
            api_key=api_key,
            custom_ip=target_ip,
            custom_port=target_port
        )
        with status_lock:  
            TASK_STATUS["query"] = {
                "running": False, 
                "msg": "Generation completed successfully", 
                "success": True,
                "context": context,
                "answer": answer
            }
    except Exception as err:
        with status_lock:  
            TASK_STATUS["query"] = {"running": False, "msg": str(err), "success": False, "context": "Process terminated.", "answer": f"Task status: {err}"}

# -------------------------------------------------------------------------
# FLASK ROUTE CONTROLLERS
# -------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/get_models", methods=["POST"])
def api_get_models():
    data = request.json or {}
    provider = data.get("provider", "").lower().strip()
    ip = data.get("ip", "localhost").replace("http://", "").replace("https://", "").strip("/")
    port = data.get("port", "").strip()
    url = f"http://{ip}:{port}"
    
    if "ollama" in provider:
        models = get_local_models(url, provider="Ollama")
    else:
        models = get_local_models(url, provider="LM Studio")
        
    return jsonify({"models": models if models else ["No models loaded"]})

@app.route("/api/check_file", methods=["POST"])
def api_check_file():
    import chromadb
    data = request.json or {}
    file_name = data.get("file_name", "")
    start_page = str(data.get("start_page", "")).strip()
    end_page = str(data.get("end_page", "")).strip()
    
    if not file_name:
        return jsonify({"error": "Invalid file name"}), 400
        
    file_base_name = os.path.splitext(file_name)[0]
    unique_seed = f"{file_base_name}_{start_page}_{end_page}"
    base_doc_id = hashlib.md5(unique_seed.encode('utf-8')).hexdigest()
    
    db_client = chromadb.PersistentClient(path=CHROMADB_DIR)
    existing_collections = [c.name for c in db_client.list_collections()]
    
    # 同時檢查 OCR 版與 VLM 版是否存在
    has_ocr = f"collection_{base_doc_id}_ocr" in existing_collections
    has_vlm = f"collection_{base_doc_id}_vlm" in existing_collections
    
    return jsonify({
        "base_doc_id": base_doc_id,
        "is_duplicate": has_ocr or has_vlm,  # 任一存在即視為已有快取
        "has_ocr": has_ocr,
        "has_vlm": has_vlm,
        "seed_info": unique_seed
    })

@app.route("/api/trigger_ocr", methods=["POST"])
def api_trigger_ocr():
    global TASK_STATUS, ACTIVE_CANCELLATIONS
    
    with status_lock:
        if TASK_STATUS["ocr"]["running"]:
            return jsonify({"error": "OCR task is already running"}), 400
        TASK_STATUS["ocr"] = {"running": True, "msg": "OCR finished...", "success": True}
        
    if 'file' not in request.files:
        with status_lock:
            TASK_STATUS["ocr"] = {"running": False, "msg": "No file provided", "success": False}
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    
    doc_id = request.form.get("doc_id", "")
    start_page = request.form.get("start_page", "")
    end_page = request.form.get("end_page", "")
    
    start_val = int(start_page) if start_page.isdigit() else None
    end_val = int(end_page) if end_page.isdigit() else None
    
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    pdf_path = os.path.join(RAW_DATA_DIR, file.filename)
    file.save(pdf_path)
    
    ACTIVE_CANCELLATIONS["ocr"] = TaskCancellation()
    
    t = threading.Thread(
        target=background_ocr_worker, 
        args=(pdf_path, doc_id, start_val, end_val, ACTIVE_CANCELLATIONS["ocr"])
    )
    t.start()
    return jsonify({"status": "started"})

@app.route("/api/trigger_vlm", methods=["POST"])
def api_trigger_vlm():
    global TASK_STATUS, ACTIVE_CANCELLATIONS
    
    with status_lock:
        if TASK_STATUS["vlm"]["running"]:
            return jsonify({"error": "VLM task is already running"}), 400
        TASK_STATUS["vlm"] = {"running": True, "msg": "Reconstructing layout via VLM...", "success": True}
        
    data = request.json or {}

    doc_id = data.get("doc_id", "")
    provider = data.get("provider", "")
    model_name = data.get("model_name", "")
    target_ip = data.get("ip", "localhost")
    target_port = data.get("port", "")
    
    ACTIVE_CANCELLATIONS["vlm"] = TaskCancellation()
    
    t = threading.Thread(
        target=background_vlm_worker,
        args=(doc_id, provider, model_name, target_ip, target_port, ACTIVE_CANCELLATIONS["vlm"])
    )
    t.start()
    return jsonify({"status": "started"})

@app.route("/api/trigger_query", methods=["POST"])
def api_trigger_query():
    global TASK_STATUS, ACTIVE_CANCELLATIONS
    
    # 檢查是否有其他查詢任務正在執行，以維護執行緒安全
    with status_lock:
        if TASK_STATUS["query"]["running"]:
            return jsonify({"error": "A query task is already running in the background."}), 400

    data = request.json or {}
    user_query = data.get("query", "").strip()
    target_id = data.get("doc_id", "").strip()  # 前端傳入的純淨 base_doc_id (例如：MD5 雜湊值)
    
    # 新增：接收前端傳入的快取製程類型，預設為 "ocr"（向下相容）
    process_type = data.get("process_type", "ocr").strip().lower()

    # 提取 LLM 模型配置參數
    provider = data.get("provider", "ollama")
    model_name = data.get("model_name", "")
    api_key = data.get("api_key", "")
    target_ip = data.get("ip", "127.0.0.1")
    target_port = data.get("port", "11434")

    if not user_query:
        return jsonify({"error": "Query text cannot be empty"}), 400
    if not target_id:
        return jsonify({"error": "Missing valid context database text reference hash identifier"}), 400

    # 防重複後綴：若前端傳入的 doc_id 已帶有 _ocr / _vlm 後綴就直接使用，否則補上
    known_suffixes = ("_ocr", "_vlm")
    if target_id.endswith(known_suffixes):
        db_target_id = target_id
    else:
        db_target_id = f"{target_id}_{process_type}"

    # 初始化任務取消憑證與狀態
    with status_lock:
        ACTIVE_CANCELLATIONS["query"] = TaskCancellation()
        TASK_STATUS["query"] = {
            "running": True,
            "step": f"Initializing structural context retrieval via vector index collection ({process_type.upper()})...",
            "progress": 0,
            "answer": ""
        }

    # 啟動背景執行緒處理檢索與模型生成
    # 注意：我們將 db_target_id (帶有後綴的識別碼) 傳入背景工作函數
    t = threading.Thread(
        target=background_query_worker,
        args=(user_query, db_target_id, provider, model_name, api_key, target_ip, target_port, ACTIVE_CANCELLATIONS["query"])
    )
    t.start()

    return jsonify({
        "status": "started", 
        "msg": f"Query process successfully targeted at collection: {db_target_id}"
    })

@app.route("/api/save_answer_md", methods=["POST"])
def api_save_answer_md():
    """
    將 AI 生成的問答解答另存為 Markdown 檔案，並儲存至 RAW_DATA_DIR 中
    """
    data = request.json or {}
    target_id = data.get("doc_id", "").strip()
    user_query = data.get("query", "").strip()
    ai_answer = data.get("answer", "").strip()
    orig_name = data.get("orig_name", "knowledge_base").strip()

    if not user_query or not ai_answer:
        return jsonify({"error": "缺少問題或 AI 回答內容，無法匯出"}), 400

    try:
        # 1. 建立儲存目錄（沿用系統既有的 RAW_DATA_DIR）
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        
        # 2. 決定 Markdown 檔案名稱（加上時間戳記避免重複）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 清除檔案名稱不合法字元
        clean_name = "".join(c for c in orig_name if c.isalnum() or c in ("_", "-")).rstrip()
        if not clean_name:
            clean_name = "QA_Export"
            
        md_file_name = f"QA_{clean_name}_{timestamp}.md"
        md_file_path = os.path.join(RAW_DATA_DIR, md_file_name)

        # 3. 組合 Markdown 內容結構
        md_content = f"""# AI 知識庫檢索最佳解答

- **來源文件識別碼**: `{target_id}`
- **產生時間**: {time.strftime("%Y-%m-%d %H:%M:%S")}

---

## ❓ 使用者提問
> {user_query}

---

## 🤖 AI 最佳解答

{ai_answer}

---
*檔案由本地 RAG 知識庫系統自動生成*
"""

        # 4. 寫入檔案
        with open(md_file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        return jsonify({
            "success": True, 
            "msg": f"Markdown 檔案已成功另存至系統目錄！",
            "file_name": md_file_name,
            "file_path": md_file_path
        })

    except Exception as e:
        return jsonify({"error": f"另存 Markdown 檔案失敗: {str(e)}"}), 500

@app.route("/api/cancel/<task_type>", methods=["POST"])
def api_cancel_task(task_type):
    global ACTIVE_CANCELLATIONS
    if task_type in ACTIVE_CANCELLATIONS:
        ACTIVE_CANCELLATIONS[task_type].is_running = False
        return jsonify({"status": f"{task_type} cancel signal sent"})
    return jsonify({"error": "Unknown task type"}), 400

_LAST_TRACKED_STATUS = {}

@app.route("/api/task_status/<task_type>", methods=["GET"])
def api_task_status(task_type):
    """Returns task status and logs changes to file and console only when mutated."""
    global _LAST_TRACKED_STATUS
    
    status_data = TASK_STATUS.get(task_type, {"running": False, "msg": "Unknown"})
    
    current_running = status_data.get("running", False)
    current_msg = status_data.get("msg", "")
    current_status_str = f"{current_running}_{current_msg}"
    
    if _LAST_TRACKED_STATUS.get(task_type) != current_status_str:
        _LAST_TRACKED_STATUS[task_type] = current_status_str
        
        status_icon = "🔄" if current_running else "✅"
        if any(x in current_msg.lower() for x in ["cancel", "fail", "error", "warning"]):
            status_icon = "🛑"
            
        log_message = f"[{status_icon} Status Changed] Module: [{task_type.upper()}] | Running: {current_running} | Message: {current_msg}"
        app_logger.info(log_message)
        
    return jsonify(status_data)

@app.route("/api/inspect_chunks", methods=["POST"])
def api_inspect_chunks():
    data = request.json or {}
    target_id = data.get("doc_id", "").strip()
    if not target_id:
        return jsonify({"error": "Missing doc_id"}), 400
        
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMADB_DIR)
        collection_name = f"collection_{target_id}"
        
        existing = [c.name for c in client.list_collections()]
        if collection_name not in existing:
            return jsonify({"error": "Collection target ID not found."}), 400
            
        collection = client.get_collection(name=collection_name)
        cached_data = collection.get(include=["documents", "metadatas"])
        
        chunks_list = []
        if cached_data and cached_data["ids"]:
            for idx, c_id in enumerate(cached_data["ids"]):
                meta = cached_data["metadatas"][idx] if cached_data["metadatas"] else {}
                chunks_list.append({
                    "index": idx,
                    "id": c_id,
                    "page": meta.get("page", "?"),
                    "type": meta.get("type", "unknown"),
                    "content": cached_data["documents"][idx]
                })
        return jsonify({"chunks": chunks_list, "target_id": target_id})
    except Exception as e:
        return jsonify({"error": f"Exception occurred during chunk parsing: {e}"}), 500

@app.route("/api/list_databases")
def api_list_databases():
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMADB_DIR)
        collections = client.list_collections()
        
        notes_data = {}
        if os.path.exists(NOTES_FILE):
            try:
                with open(NOTES_FILE, "r", encoding="utf-8") as f:
                    notes_data = json.load(f)
            except Exception:
                notes_data = {}
        
        db_list = []
        for idx, col in enumerate(collections):
            if not col.name.startswith("collection_"):
                continue
                
            total_chunks = col.count()
            orig_name = "未知檔案"
            page_desc = "未知頁碼"
            
            col_data = col.get(limit=1)
            if col_data and col_data["metadatas"]:
                for m in col_data["metadatas"]:
                    if m and "source" in m:
                        orig_name = m["source"]
                    if m and "start_page" in m and "end_page" in m:
                        page_desc = f"Page {m['start_page']} ~ Page {m['end_page']}"
                        break
                        
            clean_doc_id = col.name.replace("collection_", "")
            saved_notes = notes_data.get(clean_doc_id, "")
            if not saved_notes:
                col_meta = col.metadata if col.metadata else {}
                saved_notes = col_meta.get("notes", "")

            db_list.append({
                "index": idx + 1,
                "orig_name": orig_name,
                "page_desc": page_desc,
                "doc_id": clean_doc_id,
                "total_chunks": total_chunks,
                "notes": saved_notes  
            })
        return jsonify({"databases": db_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/update_database_notes", methods=["POST"])
def api_update_database_notes():
    data = request.json or {}
    target_doc_id = data.get("doc_id", "").strip()
    notes_content = data.get("notes", "").strip()
    if not target_doc_id:
        return jsonify({"error": "Missing target identifier"}), 400
        
    try:
        notes_data = {}
        if os.path.exists(NOTES_FILE):
            try:
                with open(NOTES_FILE, "r", encoding="utf-8") as f:
                    notes_data = json.load(f)
            except Exception:
                notes_data = {}
                
        notes_data[target_doc_id] = notes_content
        
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(notes_data, f, ensure_ascii=False, indent=4)
            
        return jsonify({"success": True, "msg": "備註更新成功"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/delete_database", methods=["POST"])
def api_delete_database():
    data = request.json or {}
    target_doc_id = data.get("doc_id", "").strip()
    if not target_doc_id:
        return jsonify({"error": "Missing target identifier"}), 400
        
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMADB_DIR)
        
        collection_name = f"collection_{target_doc_id}"
        
        # 1. 執行刪除
        client.delete_collection(name=collection_name)
        
        # 2. 安全地將記憶體資料 Flush 寫回實體硬碟 (修正 client._system.stop() 造成的永久關閉問題)
        # 透過觸發內部的唯一的資料庫密鑰同步，強迫 SQLite 釋放 WAL 鎖定並寫入
        if hasattr(client, "_producer") and hasattr(client._producer, "flush"):
            client._producer.flush()
            
        # 3. 同步清理 database_notes.json 中的備註
        if os.path.exists(NOTES_FILE):
            try:
                with open(NOTES_FILE, "r", encoding="utf-8") as f:
                    notes_data = json.load(f)
                if target_doc_id in notes_data:
                    del notes_data[target_doc_id]
                    with open(NOTES_FILE, "w", encoding="utf-8") as f:
                        json.dump(notes_data, f, ensure_ascii=False, indent=4)
            except Exception as note_err:
                app_logger.warning(f"[Warning] Failed to clean up notes for {target_doc_id}: {note_err}")

        return jsonify({"success": True, "msg": f"Collection {collection_name} deleted and physical sync completed."})
        
    except Exception as e:
        return jsonify({"error": f"Deletion failed: {e}"}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)