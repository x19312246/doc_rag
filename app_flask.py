"""主應用程式網頁界面與邏輯控制，使用 Flask 框架實現"""
import os
import sys
# import time
import hashlib
import threading
from flask import Flask, render_template, request, jsonify

import config.settings  # 確保設定被載入，尤其是路徑相關的常數

## 💡 強制 Embedding 套件進入完全離線模式，避免連線 Hugging Face Hub 卡住
#os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
#os.environ["TRANSFORMERS_OFFLINE"] = "1"
#os.environ["HF_DATASETS_OFFLINE"] = "1"

# 處理環境路徑
if getattr(sys, 'frozen', False):
    project_root = sys._MEIPASS
else:
    project_root = os.path.dirname(os.path.abspath(__file__))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

# 載入自訂基礎模組
from indexer.ocr_loader import extract_pdf_pages_info, convert_pages_to_chunks, reconstruct_pages_via_vlm
from config.settings import RAW_DATA_DIR, CHROMADB_DIR
from indexer.indexer import build_vector_index
from retriever.retriever import execute_rag_retrieval
from model.llm import query_llm, get_local_models

app = Flask(__name__, template_folder="templates", static_folder="static")

# 全局狀態追蹤字典，用來取代 QThread 的 status/is_running 控制
TASK_STATUS = {
    "ocr": {"running": False, "msg": "閒置", "success": True},
    "vlm": {"running": False, "msg": "閒置", "success": True},
    "query": {"running": False, "msg": "閒置", "success": True},
}

# 模擬中止訊號的物件
class TaskCancellation:
    def __init__(self):
        self.is_running = True

ACTIVE_CANCELLATIONS = {
    "ocr": TaskCancellation(),
    "vlm": TaskCancellation(),
    "query": TaskCancellation()
}

# -------------------------------------------------------------------------
# 後端執行緒工作函式 (原 Worker 邏輯)
# -------------------------------------------------------------------------

def background_ocr_worker(pdf_path, current_generated_id, start_page, end_page, cancel_token):
    global TASK_STATUS
    try:
        file_name = os.path.basename(pdf_path)
        file_base_name = os.path.splitext(file_name)[0]
        
        save_path = os.path.join(RAW_DATA_DIR, file_name)
        # 如果檔案不是直接上傳而是路徑，進行複製
        if pdf_path != save_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f_in, open(save_path, "wb") as f_out:
                f_out.write(f_in.read())
                
        pages_info = extract_pdf_pages_info(
            save_path, 
            dpi=200, 
            start_page=start_page, 
            end_page=end_page,
            worker_thread=cancel_token  # 傳入包含 is_running 的物件供底層檢查
        )
        
        if not cancel_token.is_running:
            TASK_STATUS["ocr"] = {"running": False, "msg": "Task cancelled by user.", "success": False}
            return
            
        chunks = convert_pages_to_chunks(
            pages_info, 
            source_name=file_base_name,
            start_page=start_page,
            end_page=end_page
        )
        total_inserted = build_vector_index(chunks, current_generated_id)
        
        TASK_STATUS["ocr"] = {"running": False, "msg": f"處理完成！共計變更/寫入 {total_inserted} 個區塊數據。", "success": True}
    except Exception as err:
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
            TASK_STATUS["vlm"] = {"running": False, "msg": "🛑 VLM 視覺萃取已由使用者中止。", "success": False}
            return
        if not new_chunks:
            TASK_STATUS["vlm"] = {"running": False, "msg": "No image source assets found or VLM response was blank.", "success": False}
            return
            
        total_inserted = build_vector_index(new_chunks, target_doc_id)
        TASK_STATUS["vlm"] = {"running": False, "msg": f"處理完成！共計變更/寫入 {total_inserted} 個區塊數據。", "success": True}
    except Exception as err:
        TASK_STATUS["vlm"] = {"running": False, "msg": str(err), "success": False}

def background_query_worker(user_query, target_id, provider, model_name, api_key, target_ip, target_port, cancel_token):
    global TASK_STATUS
    try:
        context = execute_rag_retrieval(user_query, target_id)
        
        if not cancel_token.is_running:
            TASK_STATUS["query"] = {"running": False, "msg": "Task cancelled by user.", "success": False}
            return
            
        # 2. 建立精確引導的繁體中文提示詞（Prompt）
        full_prompt = f"""你是一個專業的本地知識庫AI助手。請嚴格根據以下提供的【參考文本】來精準回答使用者的問題。
如果參考文本中找不到答案，請委婉告知「無法從目前文件中找到相關解答」，切勿編造事實。

【參考文本】：
{context}

=========================================

【使用者的問題】：
{user_query}

請提供條理清晰的繁體中文回答："""
        
        provider_mapping = {
            "本地 lmstudio": "LM Studio 本地端",
            "遠端 ollama": "Ollama 遠端/本地",
            "線上 Groq": "Groq"
        }
        p_val = provider_mapping.get(provider, "Groq")
        
        answer = query_llm(
            prompt=full_prompt,
            provider=p_val,
            model_name=model_name,
            api_key=api_key,
            custom_ip=target_ip,
            custom_port=target_port
        )
        TASK_STATUS["query"] = {
            "running": False, 
            "msg": "生成成功", 
            "success": True,
            "context": context,
            "answer": answer
        }
    except Exception as err:
        TASK_STATUS["query"] = {"running": False, "msg": str(err), "success": False, "context": "Process terminated.", "answer": f"Task status: {err}"}

# -------------------------------------------------------------------------
# Flask 路由控制器 (網頁 API 接口)
# -------------------------------------------------------------------------

@app.route("/")
def index():
    """主網頁畫面渲染"""
    return render_template("index.html")

@app.route("/api/get_models", methods=["POST"])
def api_get_models():
    """對應 action_auto_refresh_vlm_models 與 action_async_fetch_provider_models"""
    data = request.json or {}
    provider = data.get("provider", "")
    ip = data.get("ip", "localhost").replace("http://", "").replace("https://", "").strip("/")
    port = data.get("port", "").strip()
    url = f"http://{ip}:{port}"
    
    if "ollama" in provider:
        models = get_local_models(url, provider="Ollama")
    else:
        models = get_local_models(url, provider="LM Studio")
        
    return jsonify({"models": models if models else ["qwen2-vl-7b-instruct"]})

@app.route("/api/check_file", methods=["POST"])
def api_check_file():
    """對應 action_select_pdf 選擇檔案後的重複文件/ID 產算偵測機制"""
    import chromadb
    file_name = request.json.get("file_name", "")
    if not file_name:
        return jsonify({"error": "無效的檔名"}), 400
        
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
    """對應 action_trigger_ocr_pipeline"""
    global TASK_STATUS, ACTIVE_CANCELLATIONS
    if TASK_STATUS["ocr"]["running"]:
        return jsonify({"error": "已有 OCR 任務正在執行中"}), 400
        
    # 網頁端檔案處理
    if 'file' not in request.files:
        return jsonify({"error": "未提供檔案"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未選擇檔案"}), 400
        
    doc_id = request.form.get("doc_id", "")
    start_page = request.form.get("start_page", "")
    end_page = request.form.get("end_page", "")
    
    start_val = int(start_page) if start_page.isdigit() else None
    end_val = int(end_page) if end_page.isdigit() else None
    
    # 確保儲存路徑存在並寫入
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    pdf_path = os.path.join(RAW_DATA_DIR, file.filename)
    file.save(pdf_path)
    
    # 初始化執行緒狀態與取消權杖
    TASK_STATUS["ocr"] = {"running": True, "msg": "正在進行深度萃取...", "success": True}
    ACTIVE_CANCELLATIONS["ocr"] = TaskCancellation()
    
    t = threading.Thread(
        target=background_ocr_worker, 
        args=(pdf_path, doc_id, start_val, end_val, ACTIVE_CANCELLATIONS["ocr"])
    )
    t.start()
    return jsonify({"status": "started"})

@app.route("/api/trigger_vlm", methods=["POST"])
def api_trigger_vlm():
    """對應 action_trigger_vlm_reconstruct"""
    global TASK_STATUS, ACTIVE_CANCELLATIONS
    if TASK_STATUS["vlm"]["running"]:
        return jsonify({"error": "已有 VLM 任務正在執行中"}), 400
        
    data = request.json or {}
    doc_id = data.get("doc_id", "")
    provider = data.get("provider", "")
    model_name = data.get("model_name", "")
    target_ip = data.get("ip", "localhost")
    target_port = data.get("port", "")
    
    TASK_STATUS["vlm"] = {"running": True, "msg": "正在使用 VLM 進行視覺版面重建...", "success": True}
    ACTIVE_CANCELLATIONS["vlm"] = TaskCancellation()
    
    t = threading.Thread(
        target=background_vlm_worker,
        args=(doc_id, provider, model_name, target_ip, target_port, ACTIVE_CANCELLATIONS["vlm"])
    )
    t.start()
    return jsonify({"status": "started"})

@app.route("/api/trigger_query", methods=["POST"])
def api_trigger_query():
    """對應 action_execute_rag_query_pipeline"""
    global TASK_STATUS, ACTIVE_CANCELLATIONS
    if TASK_STATUS["query"]["running"]:
        return jsonify({"error": "已有查詢生成任務正在執行中"}), 400
        
    data = request.json or {}
    user_query = data.get("query", "")
    target_id = data.get("doc_id", "").strip()
    provider = data.get("provider", "")
    model_name = data.get("model_name", "")
    api_key = data.get("api_key", "")
    target_ip = data.get("ip", "localhost")
    target_port = data.get("port", "11434")
    
    if not target_id:
        return jsonify({"error": "請先輸入有效的文件識別碼。"}), 400
        
    TASK_STATUS["query"] = {"running": True, "msg": "正在檢索向量庫並調用大模型解答...", "success": True}
    ACTIVE_CANCELLATIONS["query"] = TaskCancellation()
    
    t = threading.Thread(
        target=background_query_worker,
        args=(user_query, target_id, provider, model_name, api_key, target_ip, target_port, ACTIVE_CANCELLATIONS["query"])
    )
    t.start()
    return jsonify({"status": "started"})

@app.route("/api/cancel/<task_type>", methods=["POST"])
def api_cancel_task(task_type):
    """對應 action_cancel_ocr 與 action_cancel_query 中止按鈕功能"""
    global ACTIVE_CANCELLATIONS
    if task_type in ACTIVE_CANCELLATIONS:
        ACTIVE_CANCELLATIONS[task_type].is_running = False
        return jsonify({"status": f"{task_type} cancel signal sent"})
    return jsonify({"error": "未知任務類型"}), 400

@app.route("/api/task_status/<task_type>", methods=["GET"])
def api_task_status(task_type):
    """前端輪詢執行緒狀態的 API 節點"""
    return jsonify(TASK_STATUS.get(task_type, {"running": False, "msg": "未知"}))

@app.route("/api/inspect_chunks", methods=["POST"])
def api_inspect_chunks():
    """對應 action_load_database_chunks 讀取底層 Chunks 列表"""
    data = request.json or {}
    target_id = data.get("doc_id", "").strip()
    if not target_id:
        return jsonify({"error": "請輸入 doc_id"}), 400
        
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMADB_DIR)
        collection_name = f"collection_{target_id}"
        
        existing = [c.name for c in client.list_collections()]
        if collection_name not in existing:
            return jsonify({"error": "資料庫中找不到該識別碼快取。"}), 400
            
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
        return jsonify({"error": f"讀取 Chunk 發生異常: {e}"}), 500

@app.route("/api/list_databases", methods=["GET"])
def api_list_databases():
    """對應 refresh_cached_databases 取得系統內所有既有快取列表"""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMADB_DIR)
        collections = client.list_collections()
        
        db_list = []
        for idx, col in enumerate(collections):
            col_data = col.get(include=["metadatas"])
            total_chunks = len(col_data["ids"]) if col_data and "ids" in col_data else 0
            
            orig_name = "N/A"
            page_desc = "全書範圍"
            
            if col_data and col_data["metadatas"]:
                for m in col_data["metadatas"]:
                    if m and "source" in m:
                        orig_name = m["source"]
                    if m and "start_page" in m and "end_page" in m:
                        page_desc = f"第 {m['start_page']} 頁 ~ 第 {m['end_page']} 頁"
                        break
                        
            clean_doc_id = col.name.replace("collection_", "")
            db_list.append({
                "index": idx + 1,
                "orig_name": orig_name,
                "page_desc": page_desc,
                "doc_id": clean_doc_id,
                "total_chunks": total_chunks
            })
        return jsonify({"databases": db_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/delete_database", methods=["POST"])
def api_delete_database():
    """對應 action_delete_selected_collection 刪除指定的快取資料庫"""
    data = request.json or {}
    target_doc_id = data.get("doc_id", "").strip()
    if not target_doc_id:
        return jsonify({"error": "未提供識別碼"}), 400
        
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMADB_DIR)
        client.delete_collection(name=f"collection_{target_doc_id}")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": f"刪除失敗: {e}"}), 500

if __name__ == "__main__":
    # 啟動 Flask 伺服器本地端網頁
    app.run(host="127.0.0.1", port=5000, debug=True)