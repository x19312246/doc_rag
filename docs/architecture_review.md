# 專案架構點評

> 評估日期：2026-06-09  
> 評估範圍：`app_flask.py`、`config/settings.py`、`indexer/`、`retriever/`、`model/`

---

## 優點

**關注點分離清晰**
`indexer / retriever / model / config` 四層職責明確，新人可快速定位程式碼。

**TaskCancellation Token**
以物件取代 flag 變數，傳入 worker thread 的設計乾淨，避免 global interrupt。

**狀態變更才記錄日誌**
`_LAST_TRACKED_STATUS` diff 邏輯有效抑制輪詢雜訊，是少見的細心設計。

**離線模式自動偵測**
`settings.py` 在啟動時檢查權重存在性並切換 `HF_HUB_OFFLINE`，對離線部署友善。

---

## 問題（依嚴重程度排序）

### 1. 模型每次請求重新初始化（效能重傷）

`retriever/retriever.py:11-12`

```python
def execute_rag_retrieval(user_query, target_id):
    embedding_fn = ChromaEmbeddingFunction()   # 每次查詢都 new 一個
    ...
    reranker = get_reranker()                  # 每次查詢都 new 一個
```

`SentenceTransformer` 和 `CrossEncoder` 雖然 HuggingFace 有 model cache，但 Python 物件重建仍有顯著 overhead，CPU 環境尤其明顯。這兩個物件應在 app 啟動時建立一次，以 module-level singleton 或 Flask `app_context` 持有。

---

### 2. 全域 TASK_STATUS 無鎖（競態條件）

`app_flask.py:78-82`

```python
TASK_STATUS = { "ocr": {...}, "vlm": {...}, "query": {...} }
```

Route handler 的 check-then-set 不是原子操作：

```python
if TASK_STATUS["ocr"]["running"]:     # Thread A reads False
    return error                      # Thread B also reads False
TASK_STATUS["ocr"]["running"] = True  # 兩者同時進入
```

應改用 `threading.Lock()` 或 `threading.Event` 保護。

---

### 3. Provider 正規化邏輯重複

`app_flask.py:179-185` 與 `model/llm.py:21-22` 都有獨立的 provider 字串清理邏輯，兩處行為不完全一致，日後容易出現不同步的 bug。應集中到 `model/llm.py` 一處。

---

### 4. 無 Storage 抽象層

`chromadb` 的 import 散落在 `indexer.py`、`retriever.py`、`ocr_loader.py`、`app_flask.py` 四個檔案共計超過 10 處。若日後換 pgvector 或其他後端，需逐一修改。缺少一個 `VectorStore` 介面。

---

### 5. Chunk ID 重複寫入問題

`indexer/indexer.py:18-37`

`seen_ids` 只在單次 `build_vector_index` 呼叫內去重，不跨呼叫。若對同一份 PDF 重新觸發 OCR（不刪除舊 collection），相同 ID 的 chunk 會呼叫 ChromaDB `add()`，導致報錯或靜默覆蓋，行為不可預期。應改為 `upsert`。

---

### 6. 圖片路徑硬存 metadata，路徑可移動性為零

`indexer/ocr_loader.py:228-231`

```python
"metadata": { "local_img_path": master_img }
```

絕對路徑存入 ChromaDB，專案目錄一旦移動或部署到另一台機器，VLM pipeline 全部失效。應儲存相對於 `BASE_PATH` 的相對路徑。

---

### 7. VLM 請求 `timeout=None`

`indexer/ocr_loader.py:419`

```python
res = requests.post(url, json=payload, timeout=None)
```

本意是避免大圖超時，但若 Ollama/LM Studio 服務當掉，worker thread 會永遠掛住，佔用資源且無法被 cancel token 中斷（cancel token 的 checkpoint 在這行之後）。應改為長但有限的 timeout（如 600s），並在 `requests.post` 前後加 cancel 檢查。

---

### 8. Flask 以 `debug=True` 啟動

`app_flask.py:454`

```python
app.run(host="127.0.0.1", port=5000, debug=True)
```

`debug=True` 會啟動 Werkzeug reloader，產生兩個 Python process，導致 `TASK_STATUS` 有兩份獨立副本，輪詢狀態可能對到錯誤的 process。開發外應改為 `debug=False` 或以環境變數控制。

---

## 整體評估

| 面向 | 評分 | 說明 |
|------|------|------|
| 設計意圖 | ★★★★☆ | 層次清晰，職責合理 |
| 執行品質 | ★★★☆☆ | 有數個中等以上的可靠性問題 |
| 可維護性 | ★★★☆☆ | 缺 storage 抽象，擴展成本高 |
| 效能 | ★★☆☆☆ | 模型重建是最大瓶頸，修改成本低 |

---

## 優先修復順序

| 優先序 | 問題 | 預估改動範圍 | 收益 |
|--------|------|------------|------|
| P1 | 模型 singleton | `retriever.py`、`app_flask.py` 各 ~5 行 | 查詢速度大幅提升 |
| P2 | TASK_STATUS 加鎖 | `app_flask.py` ~10 行 | 消除競態條件 |
| P3 | `upsert` 取代 `add` | `indexer.py` 1 行 | 重複索引不再報錯 |
| P4 | 圖片路徑相對化 | `ocr_loader.py` ~5 行 | 跨機器部署可用 |
| P5 | Provider 正規化集中 | `app_flask.py` 刪除、`llm.py` 調整 | 消除重複邏輯 |
| Backlog | Storage 抽象層 | 新增 `storage/` 模組 | 為 pgvector 遷移鋪路 |
