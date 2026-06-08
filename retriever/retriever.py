# import os
import re
import chromadb
from config.settings import CHROMADB_DIR
from model.embeddings import ChromaEmbeddingFunction
from model.rerank import get_reranker

def execute_rag_retrieval(user_query, target_id):
    db_client = chromadb.PersistentClient(path=CHROMADB_DIR)
    collection_name = f"collection_{target_id}"
    
    embedding_fn = ChromaEmbeddingFunction()
    collection = db_client.get_collection(name=collection_name, embedding_function=embedding_fn)
    
    # 偵測是否為範圍查詢（例如：第1頁到第3頁、第1~3頁）
    range_match = re.search(r"第\s*(\d+)\s*[頁页]\s*到\s*第\s*(\d+)\s*[頁页]|第\s*(\d+)\s*[\-~]\s*(\d+)\s*[頁页]", user_query)
    
    forced_range_context = ""
    if range_match:
        # 提取範圍頁碼
        g = range_match.groups()
        start_p = int(g[0] if g[0] else g[2])
        end_p = int(g[1] if g[1] else g[3])
        
        print(f"[Retriever] 偵測到範圍頁碼查詢：第 {start_p} 頁 到 第 {end_p} 頁。啟動保底快取全撈機制...")
        
        # 直接把這幾頁的所有區塊（包含原始 OCR 與 VLM 重塑）全部從資料庫拿出來
        all_segments = []
        for p in range(start_p, end_p + 1):
            p_res = collection.get(where={"page": p}, include=["documents", "metadatas"])
            if p_res and p_res["documents"]:
                for doc, meta in zip(p_res["documents"], p_res["metadatas"]):
                    # 標註這是哪一頁的什麼文本類型
                    t_type = "VLM視覺校正" if meta.get("type") == "vlm_text" else "原始內文"
                    all_segments.append(f"--- 【{t_type} / 第 {p} 頁】 ---\n{doc}")
        
        if all_segments:
            forced_range_context = "\n\n".join(all_segments)

    # --- 兩階段雙深度檢索 (同時進行語意交叉) ---
    query_vector = embedding_fn.embed_query(user_query)
    raw_results = collection.query(
        query_embeddings=[query_vector],
        n_results=25, 
        include=["documents", "metadatas"]
    )
    
    context_segments = []
    if raw_results and raw_results["documents"] and raw_results["documents"][0]:
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        pairs = [[user_query, doc] for doc in documents]
        
        try:
            reranker = get_reranker()
            scores = reranker.predict(pairs)
            scored_docs = sorted(zip(scores, documents, metadatas), key=lambda x: x[0], reverse=True)
            
            # 提高容納量到前 12 個黃金區塊
            top_k_results = scored_docs[:12]
            for score, doc_text, meta_data in top_k_results:
                p_info = meta_data.get("page", "?") if meta_data else "?"
                f_info = meta_data.get("source", "未知檔案") if meta_data else "未知檔案"
                segment = f"【語意精選來源: {f_info} / 第 {p_info} 頁 / Rerank得分: {score:.4f}】\n{doc_text}"
                context_segments.append(segment)
        except Exception as e:
            for i in range(min(12, len(documents))):
                doc_text = documents[i]
                meta_data = metadatas[i] if metadatas else {}
                p_info = meta_data.get("page", "?")
                segment = f"【語意精選來源 (降級排序) / 第 {p_info} 頁】\n{doc_text}"
                context_segments.append(segment)

    semantic_context = "\n\n=========================================\n\n".join(context_segments)
    
    # 🌟 最終大融合：如果使用者問的是範圍，我們把「該頁數全撈的文字」跟「語意精選的文字」合併餵給大模型
    if forced_range_context:
        return f"【系統特定頁碼範圍強制快取】\n{forced_range_context}\n\n=========================================\n\n【系統語意篩選核心內文】\n{semantic_context}"
    
    return semantic_context