import chromadb
from model.embeddings import ChromaEmbeddingFunction
from config.settings import CHROMADB_DIR

def build_vector_index(chunks, doc_id):
    client = chromadb.PersistentClient(path=CHROMADB_DIR)
    embedding_fn = ChromaEmbeddingFunction()
    
    safe_doc_id = str(doc_id).replace(" ", "_").replace("(", "-").replace(")", "-")
    collection_name = f"collection_{safe_doc_id}"
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    seen_ids = set()
    unique_ids = []
    unique_documents = []
    unique_metadatas = []
    
    for chunk in chunks:
        c_id = chunk["id"]
        if c_id in seen_ids:
            continue
        
        seen_ids.add(c_id)
        unique_ids.append(c_id)
        unique_documents.append(chunk["text"])
        unique_metadatas.append(chunk["metadata"])
        
    if unique_ids:
        collection.add(
            ids=unique_ids,
            documents=unique_documents,
            metadatas=unique_metadatas
        )
        
    return len(unique_ids)