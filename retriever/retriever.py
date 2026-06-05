import os
import re
import chromadb
from retriever.config import CHROMADB_DIR
from model.embeddings import ChromaEmbeddingFunction

def execute_rag_retrieval(user_query, target_id):
    db_client = chromadb.PersistentClient(path=CHROMADB_DIR)
    collection_name = f"collection_{target_id}"
    
    # Securely instantiate identical embedding function to match exact dimensions
    embedding_fn = ChromaEmbeddingFunction()
    collection = db_client.get_collection(name=collection_name, embedding_function=embedding_fn)
    
    page_match = re.search(r"第\s*(\d+)\s*[頁页]", user_query)
    if page_match:
        target_page = int(page_match.group(1))
        page_results = collection.get(where={"page": target_page}, include=["documents", "metadatas"])
        
        if not page_results or not page_results["documents"]:
            page_results = collection.get(where={"page": str(target_page)}, include=["documents", "metadatas"])
            
        if page_results and page_results["documents"]:
            forced_context = f"--- FORCED PAGE RETRIEVAL DETECTED: PAGE {target_page} --- \n\n"
            for doc in page_results["documents"]:
                if doc and str(doc).strip():
                    forced_context += f"{doc}\n\n"
            return forced_context

    raw_results = collection.query(query_texts=[user_query], n_results=8, include=["documents", "metadatas"])
    context_segments = []
    
    if raw_results and raw_results["documents"] and raw_results["documents"][0]:
        for i in range(len(raw_results["documents"][0])):
            doc_text = raw_results["documents"][0][i]
            meta_data = raw_results["metadatas"][0][i] if raw_results["metadatas"] else {}
            p_info = meta_data.get("page", "UNKNOWN")
            context_segments.append(f"[Source Page: {p_info}]\n{doc_text}\n--------------------")
            
    return "\n\n".join(context_segments)