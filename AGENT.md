# AGENT.md

本文件供 AI 協作開發代理（如 Gemini）閱讀，提供此專案的架構說明、開發準則與已知問題，請在提出修改建議或生成程式碼前完整閱讀。

---

## 專案定位

**本地知識庫 RAG 系統**。使用者上傳 PDF，系統進行 OCR 萃取、向量索引，再透過本地 LLM 回答問題。設計目標為**完全離線可運作**，不依賴雲端 embedding 或向量資料庫服務。

- 後端：Python / Flask
- 向量資料庫：ChromaDB（PersistentClient，底層為 SQLite + HNSWlib）
- Embedding 模型：`intfloat/multilingual-e5-large-instruct`（本地，sentence-transformers）
- Rerank 模型：`BAAI/bge-reranker-base`（本地，CrossEncoder）
- LLM 後端：Ollama、LM Studio（本地）或 Groq（雲端，需 API key）
- 前端：Jinja2 HTML 模板 + 輪詢式 JS

---

## 目錄結構

```
doc_rag/
├── app_flask.py              # Flask 主程式：路由、背景執行緒、任務狀態
├── config/
│   └── settings.py           # 所有路徑常數、模型名稱、離線模式偵測
├── indexer/
│   ├── ocr_loader.py         # PDF 萃取管線（文字、表格、圖片、VLM 重塑）
│   └── indexer.py            # ChromaDB 寫入
├── retriever/
│   └── retriever.py          # 語意搜尋 + Cross-Encoder 重排序
├── model/
│   ├── embeddings.py         # ChromaEmbeddingFunction（SentenceTransformer 封裝）
│   ├── rerank.py             # get_reranker()（CrossEncoder 封裝）
│   └── llm.py                # query_llm()：Ollama / LM Studio / Groq 路由
├── templates/                # Flask HTML 模板
├── docs/                     # 設計文件
├── model_weights/            # 本地模型權重（自動建立）
│   ├── embed/
│   └── rerank/
├── chromadb_storage/         # 向量資料庫持久化（自動建立）
├── data/
│   ├── raw/                  # 上傳的原始 PDF
│   └── processed/
│       ├── out_images/       # 頁面圖片（master 全頁 + 嵌入圖裁切）
│       └── out_tables/       # 表格 CSV
└── logs/                     # 每日滾動日誌（保留 30 天）
```

---

## 核心資料流

```
[使用者上傳 PDF]
      │
      ▼
extract_pdf_pages_info()          ← ocr_loader.py
  ├─ pdf2image：頁面轉圖
  ├─ enhance_historical_text_image()：影像強化（雙邊濾波 + 自適應二值化）
  ├─ pdfplumber：萃取數位文字層
  ├─ pytesseract：fallback OCR（文字稀少時觸發）
  └─ img2table：偵測並萃取表格
      │
      ▼
convert_pages_to_chunks()         ← ocr_loader.py
  ├─ text chunks（1000 chars / 200 overlap）
  ├─ table chunks（Markdown 格式）
  ├─ image anchor chunks（含本機圖片路徑）
  └─ global_summary chunk（每份文件一筆）
      │
      ▼
build_vector_index()              ← indexer.py
  └─ ChromaDB collection_{md5}：add chunks + embedding
      │
      ▼（選配：VLM 二次重塑）
reconstruct_pages_via_vlm()       ← ocr_loader.py
  └─ 讀取 master 頁面圖片 → 送 VLM → 產生 vlm_text chunks → 追加索引

[使用者提問]
      │
      ▼
execute_rag_retrieval()           ← retriever.py
  ├─ 偵測頁碼範圍語法（中文：第N頁到第M頁）→ 強制全撈該頁所有 chunk
  ├─ embedding query → ChromaDB top-25
  └─ CrossEncoder rerank → top-12
      │
      ▼
query_llm()                       ← llm.py
  └─ 繁體中文 prompt + context → LLM → 回答
```

---

## ChromaDB Collection 命名規則

每份文件對應一個獨立 collection：

```python
collection_name = f"collection_{md5(filename_without_extension)}"
```

`doc_id`（md5 hex）由前端在 `POST /api/check_file` 時計算並傳遞給所有後續 API。

---

## Chunk Metadata 欄位

每個 chunk 的 metadata 包含以下欄位：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `page` | int | 頁碼（global_summary 為 0） |
| `source` | str | 原始檔名（不含副檔名） |
| `type` | str | `text` / `table` / `image` / `vlm_text` / `global_summary` |
| `start_page` | int | 本次索引的起始頁 |
| `end_page` | int | 本次索引的結束頁 |
| `local_img_path` | str | 對應頁面圖片的**絕對路徑**（已知問題，見下方） |

---

## 任務管理機制

長時間操作（OCR、VLM、查詢）均在背景執行緒執行：

```python
TASK_STATUS = {
    "ocr":   {"running": bool, "msg": str, "success": bool},
    "vlm":   {"running": bool, "msg": str, "success": bool},
    "query": {"running": bool, "msg": str, "success": bool, "context": str, "answer": str},
}

ACTIVE_CANCELLATIONS = {
    "ocr":   TaskCancellation(),   # .is_running = True/False
    "vlm":   TaskCancellation(),
    "query": TaskCancellation(),
}
```

- 前端以 `GET /api/task_status/<task_type>` 輪詢狀態
- 取消操作：`POST /api/cancel/<task_type>` → 設定 `cancel_token.is_running = False`
- Worker 在關鍵 checkpoint 檢查 `cancel_token.is_running`

---

## LLM Provider 識別規則

`model/llm.py` 的 `query_llm()` 接受以下 `provider` 值（小寫）：

| provider 值 | 連線方式 | 備註 |
|------------|---------|------|
| `"ollama"` | `http://{ip}:{port}/v1`（OpenAI 相容） | 預設 port 11434 |
| `"lmstudio"` | `http://{ip}:{port}/v1` | 預設 port 1234 |
| `"groq"` | `https://api.groq.com/openai/v1` | 需傳入 `api_key` |

**注意**：`app_flask.py` 的 `background_query_worker` 中有一份重複的 provider 正規化邏輯，與 `llm.py` 行為不完全一致，這是**已知技術債**，修改時應以 `llm.py` 為準並移除 `app_flask.py` 中的重複邏輯。

---

## 離線模式

`config/settings.py` 啟動時自動偵測：

```python
if embed_ready and rerank_ready:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
```

本地權重放置路徑：
- `model_weights/embed/`：`intfloat/multilingual-e5-large-instruct`
- `model_weights/rerank/`：`BAAI/bge-reranker-base`

---

## 已知問題與技術債

以下問題**尚未修復**，提出修改建議時請一併考量：

### P1｜模型物件每次查詢重建（效能）
(6/10) `retriever/retriever.py` 的 `execute_rag_retrieval()` 每次呼叫都重新 `new ChromaEmbeddingFunction()` 和 `get_reranker()`。應改為 module-level singleton，在 app 啟動時初始化一次。

### P2｜TASK_STATUS 無鎖（競態條件）
(6/10) 多個並發請求可能同時通過 `if TASK_STATUS["ocr"]["running"]` 的判斷。應加 `threading.Lock()` 保護 check-then-set 操作。

### P3｜ChromaDB `add()` 未改為 `upsert()`
(6/10) `indexer/indexer.py` 使用 `collection.add()`，重複索引同一份 PDF 時會因 ID 衝突報錯或靜默失敗。應改為 `collection.upsert()`。

### P4｜圖片絕對路徑存入 metadata
(6/10) `local_img_path` 存的是絕對路徑，跨機器部署會失效。應改存相對於 `BASE_PATH` 的相對路徑，讀取時再組合。

### P5｜VLM 請求 `timeout=None`
(6/10) `indexer/ocr_loader.py` 對 Ollama/LM Studio 的 POST 請求設 `timeout=None`，服務當機時 worker thread 永久掛住。應設為可設定的上限（建議 600s）。

### P6｜Flask `debug=True`
(6/10) `app_flask.py` 以 `debug=True` 啟動，Werkzeug reloader 會產生兩個 process，造成 `TASK_STATUS` 有兩份副本。應改為環境變數控制。

### Backlog｜缺乏 Storage 抽象層
(PPD：會改爛程式)`chromadb` 直接散落在四個模組中，難以替換後端。長期應抽出 `VectorStore` 介面。

---

## 開發準則

### 修改前必讀
1. 確認 `config/settings.py` 中的路徑常數，**不要在其他檔案硬寫路徑**。
2. 新增 LLM 呼叫一律走 `model/llm.py` 的 `query_llm()`，不要在業務邏輯中直接建立 `OpenAI` client。
3. ChromaDB 存取目前直接呼叫 client API，修改時保持與現有欄位命名（`collection_{doc_id}`、metadata key 名稱）一致。

### 命名慣例
- Doc ID：`md5(filename_without_ext)`，hex string
- Collection：`collection_{doc_id}`
- Chunk ID：`{source_name}_p{page_num}_c{idx}`（text）、`_table_t{idx}`（table）、`_img_i{idx}`（image）、`_vlm_p{page_num}_c{idx}`（vlm）、`_global_master_anchor`（summary）

### 不要做的事
- 不要在 route handler 中直接執行耗時操作，一律包進背景執行緒。
- 不要新增新的 provider 正規化邏輯，統一修改 `model/llm.py`。
- 不要將模型物件（SentenceTransformer、CrossEncoder）建立在函式內部，應為 singleton。
- 不要在 metadata 中存絕對路徑（已知問題，新程式碼不要重蹈覆轍）。

### 測試方式
本專案目前無自動化測試套件，驗證方式為：
```bash
source .venv/bin/activate
python app_flask.py
# 開啟 http://127.0.0.1:5000 手動測試
```

---

## 環境資訊

| 項目 | 值 |
|------|-----|
| Python | 3.13 |
| 虛擬環境 | `.venv/`（venv） |
| PyTorch | CPU-only（`2.12.0+cpu`） |
| 作業系統 | Linux（VirtualBox VM） |
| GPU | 無 |
| 磁碟剩餘 | 約 3–4 GB（請注意空間）|
