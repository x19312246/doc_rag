import os
import sys
import hashlib
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

# 从 config.settings 引入 CHROMADB_DIR
from config.settings import RAW_DATA_DIR, CHROMADB_DIR

# 将 NOTES_FILE 的路径直接指向 CHROMADB_DIR 内
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
    except Exception as e:
        print(f"[CRITICAL] Model weight preloading workflow encountered a failure: {e}")
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

def background_ocr_worker(pdf_path, current_generated_id, start_page, end_page, cancel_token):
    global TASK_STATUS
    try:
        file_name = os.path.basename(pdf_path)
        file_base_name = os.path.splitext(file_name)[0]
        
        save_path = os.path.join(RAW_DATA_DIR, file_name)
        if pdf_path != save_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f_in, open(save_path, "wb") as f_out:
                f_out.write(f_in.read())
                
        pages_info = extract_pdf_pages_info(
            save_path, 
            dpi=200, 
            start_page=start_page, 
            end_page=end_page,
            worker_thread=cancel_token  
        )
        
        if not cancel_token.is_running:
            with status_lock:  
                TASK_STATUS["ocr"] = {"running": False, "msg": "Task cancelled by user.", "success": False}
            return
            
        chunks = convert_pages_to_chunks(
            pages_info, 
            source_name=file_base_name,
            start_page=start_page,
            end_page=end_page
        )
        total_inserted = build_vector_index(chunks, current_generated_id)
        
        with status_lock:  
            TASK_STATUS["ocr"] = {"running": False, "msg": f"Success! Written {total_inserted} chunk blocks.", "success": True}

    except Exception as err:
        with status_lock:  
            TASK_STATUS["ocr"] = {"running": False, "msg": str(err), "success": False}

def background_vlm_worker(target_doc_id, provider, model_name, target_ip, target_port, cancel_token):
    global TASK_STATUS
    try:
        new_chunks = reconstruct_pages_via_vlm(
            target_doc_id,
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
            
        total_inserted = build_vector_index(new_chunks, target_doc_id)
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
    file_name = request.json.get("file_name", "")
    if not file_name:
        return jsonify({"error": "Invalid file name"}), 400
        
    file_base_name = os.path.splitext(file_name)[0]
    base_doc_id = hashlib.md5(file_base_name.encode('utf-8')).hexdigest()
    
    db_client = chromadb.PersistentClient(path=CHROMADB_DIR)
    existing_collections = [c.name for c in db_client.list_collections()]
    target_collection_name = f"collection_{base_doc_id}"
    
    is_duplicate = target_collection_name in existing_collections
    return jsonify({
        "base_doc_id": base_doc_id,
        "is_duplicate": is_duplicate
    })

@app.route("/api/trigger_ocr", methods=["POST"])
def api_trigger_ocr():
    global TASK_STATUS, ACTIVE_CANCELLATIONS
    
    with status_lock:
        if TASK_STATUS["ocr"]["running"]:
            return jsonify({"error": "OCR task is already running"}), 400
        TASK_STATUS["ocr"] = {"running": True, "msg": "Extracting contents...", "success": True}
        
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

    with status_lock:
        if TASK_STATUS["query"]["running"]:
            return jsonify({"error": "Query task is already running"}), 400
        TASK_STATUS["query"] = {"running": True, "msg": "Retrieving database and generating answers...", "success": True}
        
    data = request.json or {}
    user_query = data.get("query", "")
    target_id = data.get("doc_id", "").strip()
    provider = data.get("provider", "")
    model_name = data.get("model_name", "")
    api_key = data.get("api_key", "")
    target_ip = data.get("ip", "localhost")
    target_port = data.get("port", "11434")
    
    if not target_id:
        with status_lock:
            TASK_STATUS["query"] = {"running": False, "msg": "Invalid document ID provided.", "success": False}
        return jsonify({"error": "Invalid document ID provided."}), 400
        
    ACTIVE_CANCELLATIONS["query"] = TaskCancellation()
    
    t = threading.Thread(
        target=background_query_worker,
        args=(user_query, target_id, provider, model_name, api_key, target_ip, target_port, ACTIVE_CANCELLATIONS["query"])
    )
    t.start()
    return jsonify({"status": "started"})

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
        client.delete_collection(name=f"collection_{target_doc_id}")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": f"Deletion failed: {e}"}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=4999, debug=True)