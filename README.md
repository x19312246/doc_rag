# doc_rag — 本地知識庫 RAG 系統

以 PDF 文件為輸入來源，透過 OCR、向量索引與本地大型語言模型（LLM），在完全離線的環境中完成文件問答。

---

## 功能特色

- **PDF 多模態萃取**：同時提取文字、表格（Markdown 格式）與嵌入圖片
- **影像強化 OCR**：對掃描稿件進行雙邊濾波、自適應二值化等前處理，再送入 Tesseract
- **VLM 二次重塑**：可選擇性地將頁面圖片送往本地視覺語言模型（Ollama / LM Studio）重新辨識，補足 OCR 缺失
- **離線向量索引**：使用 `intfloat/multilingual-e5-large-instruct` 嵌入，ChromaDB 持久化儲存
- **Cross-Encoder 重排序**：以 `BAAI/bge-reranker-base` 對語意搜尋結果進行二次精排
- **多 LLM 後端**：支援 Ollama、LM Studio（本地）與 Groq（雲端）
- **完全離線模式**：權重下載完成後可切斷網路，系統自動偵測並鎖定離線環境

---

## 系統需求

| 項目 | 說明 |
|------|------|
| Python | 3.10 以上 |
| Tesseract OCR | 需支援 `chi_tra`（繁體中文）及 `eng` 語言包 |
| LLM 後端 | Ollama 或 LM Studio（本地），或 Groq API 金鑰（雲端） |
| GPU（選配） | 本地 embedding / rerank 模型在 GPU 上速度更快，CPU 亦可運作 |

---

## 安裝

```bash
# 1. 安裝 Python 套件
pip install -r requirements.txt

# 2. 確認 Tesseract 已安裝並包含繁體中文語言包
#    Ubuntu / Debian:
sudo apt install tesseract-ocr tesseract-ocr-chi-tra

# 3. 啟動服務
python app_flask.py
# 預設監聽 http://127.0.0.1:5000
```

---

## 離線模式設定

首次啟動時若無本地權重，系統會自動從 HuggingFace 下載。下載完成後，將權重放置於以下路徑即可完全離線運作：

```
model_weights/
├── embed/   # intfloat/multilingual-e5-large-instruct
└── rerank/  # BAAI/bge-reranker-base
```

`config/settings.py` 會自動偵測上述資料夾，若權重存在則設定 `HF_HUB_OFFLINE=1`。

---

## 使用流程

### 1. 上傳並索引 PDF

在網頁介面上傳 PDF，選擇頁碼範圍後點擊「開始 OCR」。系統將：

1. 將每頁轉換為影像並進行強化前處理
2. 以 pdfplumber 萃取數位文字層；若文字稀少則降級為 pytesseract
3. 以 img2table 偵測並萃取表格
4. 將所有內容切塊後以向量嵌入寫入 ChromaDB

### 2. VLM 視覺重塑（選配）

OCR 完成後，可進一步將頁面圖片送往視覺語言模型（如 LLaVA、Qwen2-VL），將辨識結果以 `vlm_text` 類型追加至同一集合，提升複雜版面與圖表的問答精度。

### 3. 提問

選取已索引的文件，輸入問題後送出。系統將：

1. 語意向量搜尋（top-25）
2. Cross-Encoder 重排序取前 12 個區塊
3. 組合成繁體中文提示詞送往 LLM 生成回答

> **頁碼範圍查詢**：提問中如包含「第 N 頁到第 M 頁」或「第 N～M 頁」，系統會額外強制全撈該頁範圍的所有區塊，再與語意精選結果合併。

---

## LLM 後端設定

| 後端 | Provider 識別字 | 連線位址 | 備註 |
|------|----------------|---------|------|
| Ollama | `ollama` | `http://<ip>:11434` | 視覺模型需支援 `/api/chat` 的 `images` 欄位 |
| LM Studio | `lmstudio` | `http://<ip>:1234` | VLM 需支援 OpenAI Vision 格式 |
| Groq | `groq` | 雲端（自動） | 需提供 API 金鑰 |

---

## 目錄結構

```
doc_rag/
├── app_flask.py          # Flask 服務主程式、背景工作執行緒
├── config/
│   └── settings.py       # 路徑、模型名稱、離線模式偵測
├── indexer/
│   ├── ocr_loader.py     # PDF 萃取、影像強化、VLM 重塑管線
│   └── indexer.py        # ChromaDB 寫入
├── retriever/
│   └── retriever.py      # 語意搜尋 + Cross-Encoder 重排序
├── model/
│   ├── embeddings.py     # SentenceTransformer 嵌入封裝
│   ├── rerank.py         # CrossEncoder 重排序封裝
│   └── llm.py            # Ollama / LM Studio / Groq 路由
├── templates/            # Flask HTML 模板
├── model_weights/        # 本地模型權重（自動建立）
├── chromadb_storage/     # 向量資料庫持久化（自動建立）
├── data/
│   ├── raw/              # 上傳的原始 PDF
│   └── processed/        # 萃取的圖片與表格 CSV
└── logs/                 # 每日滾動日誌
```

---

## 日誌

應用程式日誌寫入 `logs/rag_system_daily.log`，保留 30 天。輪詢端點（`/api/task_status/*`）的存取記錄會被過濾，不寫入日誌也不輸出至終端機，避免雜訊干擾。
