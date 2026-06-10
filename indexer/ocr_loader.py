import os
import hashlib
import pdfplumber
import numpy as np
import cv2
from PIL import Image as PILImage
import pytesseract
from pdf2image import convert_from_path
from img2table.document import Image as ImgDoc
from img2table.ocr import TesseractOCR
import requests
import base64
import threading
import time
from config.settings import OUTPUT_TABLES_DIR, BASE_PATH

OUTPUT_IMAGES_DIR = os.path.join(os.path.dirname(OUTPUT_TABLES_DIR), "out_images")
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

def group_words_to_lines(words, line_tol=3):
    if not words:
        return ""
    words_sorted = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    lines = []
    cur_top = None
    cur_words = []
    for w in words_sorted:
        if cur_top is None:
            cur_top = w["top"]
            cur_words = [w["text"]]
        elif abs(w["top"] - cur_top) <= line_tol:
            cur_words.append(w["text"])
        else:
            lines.append(" ".join(cur_words))
            cur_top = w["top"]
            cur_words = [w["text"]]
    if cur_words:
        lines.append(" ".join(cur_words))
    return "\n".join(lines)

def enhance_historical_text_image(pil_image):
    open_cv_image = np.array(pil_image)
    if len(open_cv_image.shape) == 3:
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2GRAY)
    else:
        gray = open_cv_image

    filtered = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    binary = cv2.adaptiveThreshold(
        filtered, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        25, 
        15
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return PILImage.fromarray(cleaned)

def extract_pdf_pages_info(pdf_path, dpi=200, start_page=None, end_page=None, worker_thread=None):
    pages_info = []
    ocr_engine = TesseractOCR(n_threads=4, lang="chi_tra+eng")
    
    print(f"[Engine] Targeted PDF: {os.path.basename(pdf_path)}")
    
    with pdfplumber.open(pdf_path) as pdf:
        total_pdf_pages = len(pdf.pages)
        
    p_start = int(start_page) if start_page is not None else 1
    p_end = int(end_page) if end_page is not None else total_pdf_pages
    p_start = max(1, min(p_start, total_pdf_pages))
    p_end = max(p_start, min(p_end, total_pdf_pages))
    
    print(f"[Engine] Execution scope: Page {p_start} to Page {p_end}")
    
    for page_idx in range(p_start - 1, p_end):
        if worker_thread and not worker_thread.is_running:
            print("[Engine] Loop termination triggered by worker signal.")
            break
            
        page_num = page_idx + 1
        print(f" -> Processing Page {page_num}/{total_pdf_pages}...")
        
        master_page_img_name = f"master_page_{page_num}_{hashlib.md5(str(pdf_path).encode()).hexdigest()[:6]}.png"
        master_page_img_path = os.path.join(OUTPUT_IMAGES_DIR, master_page_img_name)
        
        try:
            pil_images = convert_from_path(
                pdf_path, 
                dpi=dpi, 
                first_page=page_num, 
                last_page=page_num
            )
            page_pil = pil_images[0] if pil_images else None
            
            if page_pil:
                page_pil = enhance_historical_text_image(page_pil)
                page_pil.save(master_page_img_path, format="PNG")
                
        except Exception as e:
            print(f"    [Warning] pdf2image conversion or enhancement failed on page {page_num}: {e}")
            page_pil = None
            master_page_img_path = ""
            
        tables_found = []
        if page_pil and master_page_img_path:
            try:
                img_doc = ImgDoc(src=master_page_img_path)
                extracted_tables = img_doc.extract_tables(
                    ocr=ocr_engine,
                    implicit_rows=True,
                    implicit_columns=True,
                    borderless_tables=True
                )
                if extracted_tables:
                    tables_found = extracted_tables
            except Exception as e:
                print(f"    [Warning] Table detection failed on page {page_num}: {e}")

        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_idx]
            words = page.extract_words()
            page_images = page.images
            
            page_text = ""
            if words:
                page_text = group_words_to_lines(words)
                
            if not page_text.strip() or len(page_text.strip()) < 10:
                if page_pil:
                    try:
                        print(f"    [Strong OCR Fallback] Scanning preprocessed image layer via pytesseract...")
                        page_text = pytesseract.image_to_string(page_pil, lang="chi_tra+eng")
                    except Exception as ocr_err:
                        print(f"    [Error] Pytesseract execution block failed: {ocr_err}")
                        page_text = "[OCR Failed Image Layer]"
                else:
                    page_text = "[Empty Digital Layer]"
            
            images_data = []
            try:
                for img_idx, img_obj in enumerate(page_images):
                    if worker_thread and not worker_thread.is_running:
                        break
                    if img_obj["width"] < 40 or img_obj["height"] < 40:
                        continue
                        
                    cropped_img = page.within_bbox((img_obj["x0"], img_obj["top"], img_obj["x1"], img_obj["bottom"])).to_image(resolution=150)
                    img_file_name = f"img_p{page_num}_{img_idx}_{hashlib.md5(str(img_obj).encode()).hexdigest()[:6]}.png"
                    img_file_path = os.path.join(OUTPUT_IMAGES_DIR, img_file_name)
                    cropped_img.save(img_file_path, format="PNG")
                    
                    images_data.append({
                        "image_idx": img_idx,
                        "file_path": img_file_path,
                        "file_name": img_file_name,
                        "bbox": (float(img_obj["x0"]), float(img_obj["top"]), float(img_obj["x1"]), float(img_obj["bottom"]))
                    })
            except Exception as img_err:
                print(f"    [Warning] Image extraction loop skipped: {img_err}")
                
            tables_data = []
            for t_idx, tbl in enumerate(tables_found):
                try:
                    df = tbl.df
                    if df is not None and not df.empty:
                        csv_name = f"table_p{page_num}_t{t_idx}.csv"
                        csv_path = os.path.join(OUTPUT_TABLES_DIR, csv_name)
                        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                        tables_data.append({
                            "table_idx": t_idx,
                            "markdown": df.to_markdown(index=False),
                            "csv_path": csv_path
                        })
                except Exception as t_err:
                    print(f"    [Warning] CSV export layout skipped: {t_err}")
            
            pages_info.append({
                "page_num": page_num,
                "text": page_text,
                "tables": tables_data,
                "images": images_data,
                "master_page_image": master_page_img_path
            })
            page.flush_cache()
            
    return pages_info

def convert_pages_to_chunks(pages_info, source_name="", start_page=None, end_page=None):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    
    docs = []
    summary_accumulator = []
    
    meta_start = int(start_page) if start_page is not None else 1
    meta_end = int(end_page) if end_page is not None else (pages_info[-1]["page_num"] if pages_info else 1)
    
    for p_data in pages_info:
        page_num = p_data["page_num"]
        raw_text = p_data["text"]
        tables = p_data["tables"]
        images = p_data["images"]
        master_img = p_data.get("master_page_image", "")
        
        snippet = raw_text.strip()[:30].replace("\n", " ")
        summary_accumulator.append(f"- Page {page_num} Snippet: {snippet}...")
        
        if raw_text.strip():
            chunks = splitter.split_text(raw_text)
            for c_idx, chunk_txt in enumerate(chunks):
                docs.append({
                    "id": f"{source_name}_p{page_num}_c{c_idx}",
                    "text": f"【歷史文件內文】\n檔案名稱: {source_name}\n所在頁碼: 第 {page_num} 頁\n\n實質內文:\n{chunk_txt}",
                    "metadata": {
                        "page": page_num, 
                        "source": source_name, 
                        "type": "text",
                        "start_page": meta_start,
                        "end_page": meta_end,
                        "local_img_path": os.path.relpath(master_img, BASE_PATH) if master_img else ""
                    }
                })
        
        for tbl_item in tables:
            t_idx = tbl_item["table_idx"]
            md_text = tbl_item["markdown"]
            if md_text.strip():
                docs.append({
                    "id": f"{source_name}_p{page_num}_table_t{t_idx}",
                    "text": f"【歷史文件精準表格】\n檔案名稱: {source_name}\n所在頁碼: 第 {page_num} 頁 (第 {t_idx} 組表格)\n\n表格數據:\n{md_text}",
                    "metadata": {
                        "page": page_num, 
                        "source": source_name, 
                        "type": "table",
                        "start_page": meta_start,
                        "end_page": meta_end,
                        "local_img_path": os.path.relpath(master_img, BASE_PATH) if master_img else ""
                    }
                })
                
        for img_item in images:
            i_idx = img_item["image_idx"]
            f_name = img_item["file_name"]
            
            surrounding_context = raw_text.strip()[:200].replace("\n", " ")
            image_chunk_text = (
                f"【歷史文件圖片錨點】\n"
                f"檔案名稱: {source_name}\n"
                f"所在頁碼: 第 {page_num} 頁 (第 {i_idx} 張嵌入圖)\n"
                f"實體圖片快取檔名: {f_name}\n"
                f"實體圖片存放路徑: {f_name}\n"
                f"圖片所在頁面之前景文字脈絡參考:\n{surrounding_context}"
            )
            
            chosen_path = master_img if master_img else img_item["file_path"]
            
            docs.append({
                "id": f"{source_name}_p{page_num}_img_i{i_idx}",
                "text": image_chunk_text,
                "metadata": {
                    "page": page_num, 
                    "source": source_name, 
                    "type": "image",
                    "local_img_path": os.path.relpath(chosen_path, BASE_PATH) if chosen_path else "",
                    "start_page": meta_start,
                    "end_page": meta_end
                }
            })
        
    if summary_accumulator:
        docs.append({
            "id": f"{source_name}_global_master_anchor",
            "text": f"【本篇文件選定範圍全域摘要】\n檔案名稱：《{source_name}》\n\n結構節錄：\n" + "\n".join(summary_accumulator),
            "metadata": {
                "page": 0, 
                "source": source_name, 
                "type": "global_summary",
                "start_page": meta_start,
                "end_page": meta_end,
                "local_img_path": ""
            }
        })
        
    return docs

def _vlm_timeout_monitor(page_num, stop_event, worker_thread):
    """
    Background non-blocking watcher thread to alert the user when a single page 
    VLM processing block exceeds the 600-second threshold.
    """
    start_time = time.time()
    warned = False
    while not stop_event.is_set():
        if worker_thread and not worker_thread.is_running:
            break
        elapsed = time.time() - start_time
        if elapsed >= 600.0 and not warned:
            print("\n" + "="*80)
            print(f"[VLM WARNING] Page {page_num} processing has exceeded 600 seconds (10 minutes)!")
            print("              If local server resources show 0% load, the engine may be frozen.")
            print("              The user can manually trigger 'Cancel' via the UI control panel at any time.")
            print("="*80 + "\n")
            warned = True
            break
        time.sleep(2.0)

def reconstruct_pages_via_vlm(target_doc_id, provider, model_name, target_ip, target_port, worker_thread=None):
    import chromadb
    from config.settings import CHROMADB_DIR
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    print("[VLM Entry Verification]")
    print(f" -> Passed Provider: '{provider}'")
    print(f" -> Passed Model: '{model_name}'")
    print(f" -> Target Host URL: http://{target_ip}:{target_port}")
    
    provider_clean = str(provider).strip().lower()
    if "ollama" in provider_clean:
        health_url = f"http://{str(target_ip).strip()}:{str(target_port).strip()}/api/tags"
    else:
        health_url = f"http://{str(target_ip).strip()}:{str(target_port).strip()}/v1/models"
    
    try:
        print(f"[VLM Health Check] Connecting to {health_url}...")
        res = requests.get(health_url, timeout=5)
        if res.status_code != 200:
            print(f"[VLM FATAL ERROR] Service at {target_ip}:{target_port} returned {res.status_code}. Aborting.")
            return []
        print("[VLM Health Check] Service is online and responding.")
    except requests.exceptions.Timeout:
        print(f"[VLM FATAL ERROR] Connection timeout to {target_ip}:{target_port}. Service may be unresponsive. Aborting.")
        return []
    except requests.exceptions.ConnectionError as e:
        print(f"[VLM FATAL ERROR] Cannot reach {target_ip}:{target_port}. Check IP/Port or if service is running. Error: {str(e)[:100]}. Aborting.")
        return []
    except Exception as e:
        print(f"[VLM FATAL ERROR] Unexpected error during health check: {str(e)[:100]}. Aborting.")
        return []
    
    client = chromadb.PersistentClient(path=CHROMADB_DIR)
    collection_name = f"collection_{target_doc_id}"
    
    try:
        collection = client.get_collection(name=collection_name)
        existing_data = collection.get(include=["metadatas"])
    except Exception:
        return []
        
    if not existing_data or not existing_data["metadatas"]:
        return []
        
    image_tasks = {}
    source_name = "Reconstructed_Document"
    meta_start = 1
    meta_end = 1
    
    for meta in existing_data["metadatas"]:
        if meta.get("local_img_path"):
            p_num = meta.get("page", 1)
            if p_num > 0:
                rel_path = meta.get("local_img_path")
                if os.path.isabs(rel_path):
                    image_tasks[p_num] = rel_path
                else:
                    image_tasks[p_num] = os.path.normpath(os.path.join(BASE_PATH, rel_path))
                    
        if meta.get("source"):
            source_name = meta.get("source")
            
    if not image_tasks:
        print("[VLM Engine] Termination: No master full-page image reference found in metadata database.")
        return []
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    new_vlm_chunks = []
    
    for page_num in sorted(image_tasks.keys()):
        if worker_thread and not worker_thread.is_running:
            print("[VLM Reconstruct] User cancelled - stopping current batch.")
            break
            
        img_path = image_tasks[page_num]
        if not os.path.exists(img_path):
            continue
            
        print(f"[VLM Reconstruct] Routing page {page_num} via explicit target specs...")
        
        try:
            if worker_thread and not worker_thread.is_running:
                print(f"[VLM Reconstruct] Cancelled before processing page {page_num}.")
                break
            
            with open(img_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            if worker_thread and not worker_thread.is_running:
                print(f"[VLM Reconstruct] Cancelled before VLM request for page {page_num}.")
                break
            
            prompt_text = (
                "請仔細閱讀這張歷史文件圖片，將其內文轉換為高精確度的繁體中文 Markdown 格式。"
                "如果頁面中包含統計圖表、數學模型公式（如替代效果與所得效果的變動公式）或經濟學供需曲線表格，"
                "請將其精確重構為標準 Markdown 表格或純文字公式敘述。請勿添加任何個人多餘的主觀對話或說明提示。"
            )
            
            vlm_text = ""
            
            # Start background monitoring thread for 600s warning trigger
            monitor_stop = threading.Event()
            monitor_thread = threading.Thread(
                target=_vlm_timeout_monitor, 
                args=(page_num, monitor_stop, worker_thread), 
                daemon=True
            )
            monitor_thread.start()
            
            if "ollama" in provider_clean:
                url = f"http://{str(target_ip).strip()}:{str(target_port).strip()}/api/chat"
                payload = {
                    "model": model_name,
                    "messages": [{
                        "role": "user",
                        "content": prompt_text,
                        "images": [base64_image]
                    }],
                    "stream": False
                }
                try:
                    res = requests.post(url, json=payload, timeout=None)
                    if res.status_code == 200:
                        vlm_text = res.json().get("message", {}).get("content", "")
                    else:
                        print(f"[VLM ERROR] Page {page_num}: HTTP {res.status_code} - {res.text[:200]}")
                except requests.exceptions.RequestException as req_err:
                    print(f"[VLM ERROR] Page {page_num}: Network error - {str(req_err)[:200]}")
            else:
                url = f"http://{str(target_ip).strip()}:{str(target_port).strip()}/v1/chat/completions"
                payload = {
                    "model": model_name,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }],
                    "stream": False
                }
                try:
                    res = requests.post(url, json=payload, timeout=None)
                    if res.status_code == 200:
                        vlm_text = res.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                    else:
                        print(f"[VLM ERROR] Page {page_num}: HTTP {res.status_code} - {res.text[:200]}")
                except requests.exceptions.RequestException as req_err:
                    print(f"[VLM ERROR] Page {page_num}: Network error - {str(req_err)[:200]}")
            
            # Request terminated, clean up monitor thread safely
            monitor_stop.set()
            
            if worker_thread and not worker_thread.is_running:
                print(f"[VLM Reconstruct] Cancelled after VLM response for page {page_num}.")
                break
            
            if not vlm_text or not vlm_text.strip() or len(vlm_text.strip()) < 15:
                print(f"[CRITICAL ERROR] VLM layer execution returned blank on page {page_num}!")
                print(" -> Hint: Verify if your active LM Studio model supports Vision features. Pure text models will drop images silently.")
                continue
                
            if worker_thread and not worker_thread.is_running:
                print(f"[VLM Reconstruct] Cancelled before text splitting for page {page_num}.")
                break
            
            sub_chunks = splitter.split_text(vlm_text)
            for c_idx, txt in enumerate(sub_chunks):
                new_vlm_chunks.append({
                    "id": f"{source_name}_vlm_p{page_num}_c{c_idx}",
                    "text": f"【VLM視覺重塑深度內文】\n檔案名稱: {source_name}\n所在頁碼: 第 {page_num} 頁\n\n校正內文:\n{txt}",
                    "metadata": {
                        "page": page_num,
                        "source": source_name,
                        "type": "vlm_text",
                        "start_page": meta_start,
                        "end_page": meta_end,
                        "local_img_path": os.path.relpath(img_path, BASE_PATH) if not os.path.isabs(img_path) else img_path
                    }
                })
        except Exception as e:
            print(f"[Warning] VLM request processing failed on page {page_num}: {e}")
            
    return new_vlm_chunks