"""主應用程式界面與邏輯控制，使用 PyQt6 實現"""
## 引用I: 主要內建函式
import os
import sys
import time
import hashlib

# 💡 強制 Embedding 套件進入完全離線模式，避免連線 Hugging Face Hub 卡住
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

## 引用II: PyQt6 相關元件
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, 
    QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton, 
    QComboBox, QTextEdit, QTableWidget, QTableWidgetItem, 
    QProgressBar, QMessageBox, QRadioButton, QButtonGroup, QHeaderView, QListWidget
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QPoint
from PyQt6.QtGui import QFont

if getattr(sys, 'frozen', False):
    project_root = sys._MEIPASS
else:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from indexer.ocr_loader import extract_pdf_pages_info, convert_pages_to_chunks, reconstruct_pages_via_vlm
from config.settings import RAW_DATA_DIR, CHROMADB_DIR
from indexer.indexer import build_vector_index
from retriever.retriever import execute_rag_retrieval
from model.llm import query_llm, get_local_models

## 給資料庫區塊檢視校對器使用的自訂拖曳分隔線元件
class ResizeHandle(QWidget):
    """自訂的拖曳分隔線"""
    def __init__(self, target_widget, parent=None):
        super().__init__(parent)
        self.target_widget = target_widget
        self.setFixedWidth(6)  # 分隔線的寬度
        self.setCursor(Qt.CursorShape.SplitHCursor)  # 讓滑鼠變成左右調整的圖示
        self.setStyleSheet("background-color: #cbd5e1;")  # 邊框顏色
        self.dragging = False
        self.start_x = 0
        self.start_width = 0

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.start_x = event.globalPosition().x()
            self.start_width = self.target_widget.width()

    def mouseMoveEvent(self, event):
        if self.dragging:
            delta = event.globalPosition().x() - self.start_x
            new_width = max(50, self.start_width + int(delta))  # 限制最小寬度為 50px
            self.target_widget.setFixedWidth(new_width)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False


class OcrWorker(QThread):
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, pdf_path, current_generated_id, start_page=None, end_page=None):
        super().__init__()
        self.pdf_path = pdf_path
        self.current_generated_id = current_generated_id
        self.start_page = start_page
        self.end_page = end_page
        self.is_running = True 

    def run(self):
        try:
            file_name = os.path.basename(self.pdf_path)
            file_base_name = os.path.splitext(file_name)[0]
            
            save_path = os.path.join(RAW_DATA_DIR, file_name)
            with open(self.pdf_path, "rb") as f_in, open(save_path, "wb") as f_out:
                f_out.write(f_in.read())
                
            pages_info = extract_pdf_pages_info(
                save_path, 
                dpi=200, 
                start_page=self.start_page, 
                end_page=self.end_page,
                worker_thread=self
            )
            
            if not self.is_running:
                self.error_signal.emit("Task cancelled by user.")
                return
                
            chunks = convert_pages_to_chunks(
                pages_info, 
                source_name=file_base_name,
                start_page=self.start_page,
                end_page=self.end_page
            )
            total_inserted = build_vector_index(chunks, self.current_generated_id)
            
            self.finished_signal.emit(total_inserted)
        except Exception as err:
            self.error_signal.emit(str(err))

class VlmReconstructWorker(QThread):
    finished_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)

    def __init__(self, target_doc_id, provider, model_name, target_ip, target_port):
        super().__init__()
        self.target_doc_id = target_doc_id
        self.provider = provider
        self.model_name = model_name
        self.target_ip = target_ip
        self.target_port = target_port
        self.is_running = True

    def run(self):
        try:
            new_chunks = reconstruct_pages_via_vlm(
                self.target_doc_id,
                self.provider,
                self.model_name,
                self.target_ip,
                self.target_port,
                worker_thread=self
            )
            if not self.is_running:
                print("[VLM Worker] Task was cancelled by user - aborting vector indexing.")
                self.error_signal.emit("🛑 VLM 視覺萃取已由使用者中止。")
                return
            if not new_chunks:
                self.error_signal.emit("No image source assets found or VLM response was blank.")
                return
                
            total_inserted = build_vector_index(new_chunks, self.target_doc_id)
            self.finished_signal.emit(total_inserted)
        except Exception as err:
            if not self.is_running:
                print("[VLM Worker] Task was cancelled during exception handling.")
                self.error_signal.emit("🛑 VLM 視覺萃取已由使用者中止。")
            else:
                self.error_signal.emit(str(err))

class QueryWorker(QThread):
    finished_signal = pyqtSignal(str, str)
    error_signal = pyqtSignal(str)

    def __init__(self, user_query, target_id, provider, model_name, api_key, target_ip, target_port):
        super().__init__()
        self.user_query = user_query
        self.target_id = target_id
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.target_ip = target_ip
        self.target_port = target_port
        self.is_running = True

    def run(self):
        try:
            context = execute_rag_retrieval(self.user_query, self.target_id)
            
            if not self.is_running:
                self.error_signal.emit("Task cancelled by user.")
                return
                
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{self.user_query}\nAnswer in Traditional Chinese format:"
            
            provider_mapping = {
                "本地 lmstudio": "LM Studio 本地端",
                "遠端 ollama": "Ollama 遠端/本地",
                "線上 Groq": "Groq"
            }
            p_val = provider_mapping.get(self.provider, "Groq")
            
            answer = query_llm(
                prompt=full_prompt,
                provider=p_val,
                model_name=self.model_name,
                api_key=self.api_key,
                custom_ip=self.target_ip,
                custom_port=self.target_port
            )
            self.finished_signal.emit(context, answer)
        except Exception as err:
            self.error_signal.emit(str(err))

class RagQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("簡易 RAG 檢索系統 (先 bge-m3 再 vlm)")
        self.resize(1200, 850)
        
        self.selected_pdf_path = ""
        self.current_generated_id = ""
        self.ocr_thread = None
        self.vlm_thread = None
        self.query_thread = None
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        system_font = QFont("Microsoft JhengHei", 10)
        self.monospace_font_name = "Consolas"
        if sys.platform == "darwin":
            system_font = QFont("PingFang TC", 13)
            self.monospace_font_name = "Menlo"
            
        self.setFont(system_font)
        
        self.setup_upload_tab()
        self.setup_qa_tab()
        self.setup_inspect_tab()
        self.setup_manager_tab()
        self.refresh_cached_databases()

    def create_styled_label(self, text, style_str="color: #1e293b; font-weight: normal;"):
        lbl = QLabel(text)
        lbl.setStyleSheet(style_str)
        return lbl

    def setup_upload_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        file_layout = QHBoxLayout()
        self.btn_select = QPushButton("選擇 PDF 檔案...")
        self.btn_select.clicked.connect(self.action_select_pdf)
        file_layout.addWidget(self.btn_select)
        
        self.lbl_file_status = self.create_styled_label("尚未選擇任何檔案", "color: #334155; font-weight: bold;")
        file_layout.addWidget(self.lbl_file_status)
        file_layout.addStretch()
        layout.addLayout(file_layout)
        
        self.page_range_box = QWidget()
        page_range_layout = QVBoxLayout(self.page_range_box)
        page_range_layout.addWidget(self.create_styled_label("🎯 頁數擷取：大型文件 PDF 擷取頁數範圍 (空白表示擷取全檔)", "color: #0f172a; font-weight: bold;"))
        
        inputs_layout = QHBoxLayout()
        inputs_layout.addWidget(self.create_styled_label("起始頁碼:", "color: #334155;"))
        self.ent_start_page = QLineEdit()
        self.ent_start_page.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1; max-width: 80px;")
        inputs_layout.addWidget(self.ent_start_page)
        
        inputs_layout.addWidget(self.create_styled_label("結束頁碼:", "color: #334155;"))
        self.ent_end_page = QLineEdit()
        self.ent_end_page.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1; max-width: 80px;")
        inputs_layout.addWidget(self.ent_end_page)
        inputs_layout.addStretch()
        page_range_layout.addLayout(inputs_layout)
        layout.addWidget(self.page_range_box)
        self.page_range_box.setVisible(False)
        
        self.conflict_group_box = QWidget()
        conflict_layout = QVBoxLayout(self.conflict_group_box)
        self.lbl_id_msg = QLabel("")
        conflict_layout.addWidget(self.lbl_id_msg)
        
        self.radio_btn_group = QButtonGroup(self)
        self.radio_keep = QRadioButton("直接導引原 ID (沿用歷史快取，免重新辨識)")
        self.radio_overwrite = QRadioButton("蓋過舊資料 (更換全新 ID，徹底重新辨識)")
        self.radio_btn_group.addButton(self.radio_keep)
        self.radio_btn_group.addButton(self.radio_overwrite)
        conflict_layout.addWidget(self.radio_keep)
        conflict_layout.addWidget(self.radio_overwrite)
        
        id_display_layout = QHBoxLayout()
        id_display_layout.addWidget(self.create_styled_label("自動產生識別碼: "))
        self.ent_doc_id = QLineEdit()
        self.ent_doc_id.setReadOnly(True)
        self.ent_doc_id.setStyleSheet(f"background-color: #e2e8f0; color: #0f172a; font-family: {self.monospace_font_name}; border: 1px solid #cbd5e1;")
        id_display_layout.addWidget(self.ent_doc_id)
        conflict_layout.addLayout(id_display_layout)
        
        self.radio_keep.clicked.connect(self.action_toggle_conflict_strategy)
        self.radio_overwrite.clicked.connect(self.action_toggle_conflict_strategy)
        layout.addWidget(self.conflict_group_box)
        self.conflict_group_box.setVisible(False)
        
        self.t1_vlm_config_box = QWidget()
        t1_vlm_layout = QGridLayout(self.t1_vlm_config_box)
        t1_vlm_layout.setContentsMargins(0, 5, 0, 5)
        
        t1_vlm_layout.addWidget(self.create_styled_label("✨ VLM 類別:", "color: #0f172a; font-weight: bold;"), 0, 0)
        self.t1_vlm_provider_combo = QComboBox()
        self.t1_vlm_provider_combo.addItems(["本地 lmstudio", "遠端 ollama"])
        self.t1_vlm_provider_combo.setStyleSheet("color: #000000; background-color: #ffffff;")
        t1_vlm_layout.addWidget(self.t1_vlm_provider_combo, 0, 1)
        
        t1_vlm_layout.addWidget(self.create_styled_label("VLM Host IP:"), 0, 2)
        self.t1_vlm_ip_edit = QLineEdit("localhost")
        self.t1_vlm_ip_edit.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1;")
        t1_vlm_layout.addWidget(self.t1_vlm_ip_edit, 0, 3)
        
        t1_vlm_layout.addWidget(self.create_styled_label("Port:"), 0, 4)
        self.t1_vlm_port_edit = QLineEdit("1234")
        self.t1_vlm_port_edit.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1;")
        t1_vlm_layout.addWidget(self.t1_vlm_port_edit, 0, 5)
        
        btn_t1_refresh = QPushButton("🔄 刷新模型清單")
        btn_t1_refresh.clicked.connect(self.action_auto_refresh_vlm_models)
        t1_vlm_layout.addWidget(btn_t1_refresh, 0, 6)
        
        t1_vlm_layout.addWidget(self.create_styled_label("指定 VLM 模型名稱標籤:"), 1, 0)
        self.t1_vlm_model_edit = QComboBox()
        self.t1_vlm_model_edit.setStyleSheet("color: #000000; background-color: #ffffff;")
        t1_vlm_layout.addWidget(self.t1_vlm_model_edit, 1, 1, 1, 5)
        
        layout.addWidget(self.t1_vlm_config_box)
        self.t1_vlm_config_box.setVisible(False)
        
        ocr_control_layout = QHBoxLayout()
        self.btn_run_ocr = QPushButton("🚀 開始使用 bge-m3 萃取 文字/表格/圖片")
        self.btn_run_ocr.setStyleSheet("background-color: #2da44e; color: white; font-weight: bold; height: 35px;")
        self.btn_run_ocr.clicked.connect(self.action_trigger_ocr_pipeline)
        ocr_control_layout.addWidget(self.btn_run_ocr, 4)
        
        self.btn_stop_ocr = QPushButton("🛑 停止萃取")
        self.btn_stop_ocr.setStyleSheet("background-color: #b91c1c; color: white; font-weight: bold; height: 35px;")
        self.btn_stop_ocr.setEnabled(False)
        self.btn_stop_ocr.clicked.connect(self.action_cancel_ocr)
        ocr_control_layout.addWidget(self.btn_stop_ocr, 1)
        layout.addLayout(ocr_control_layout)
        
        vlm_control_layout = QHBoxLayout()
        self.btn_run_vlm_reconstruct = QPushButton("✨ 開始使用 VLM 視覺萃取 (整合匯入原向量庫)")
        self.btn_run_vlm_reconstruct.setStyleSheet("background-color: #7c3aed; color: white; font-weight: bold; height: 35px;")
        self.btn_run_vlm_reconstruct.clicked.connect(self.action_trigger_vlm_reconstruct)
        vlm_control_layout.addWidget(self.btn_run_vlm_reconstruct)
        layout.addLayout(vlm_control_layout)
        self.btn_run_vlm_reconstruct.setVisible(False)
        
        self.progress_ocr = QProgressBar()
        self.progress_ocr.setRange(0, 0)
        layout.addWidget(self.progress_ocr)
        self.progress_ocr.setVisible(False)
        
        layout.addStretch()
        self.tabs.addTab(tab, " 📁 新文件上傳與索引化 ")

    def action_auto_refresh_vlm_models(self):
        provider = self.t1_vlm_provider_combo.currentText()
        ip = self.t1_vlm_ip_edit.text().replace("http://", "").replace("https://", "").strip("/")
        port = self.t1_vlm_port_edit.text().strip()
        url = f"http://{ip}:{port}"
        
        if provider == "遠端 ollama":
            models = get_local_models(url, provider="Ollama")
        else:
            models = get_local_models(url, provider="LM Studio")
            
        self.t1_vlm_model_edit.clear()
        if models:
            self.t1_vlm_model_edit.addItems(models)
        else:
            self.t1_vlm_model_edit.addItem("qwen2-vl-7b-instruct")

    def action_select_pdf(self):
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(self, "選擇 PDF 檔案", "", "PDF Files (*.pdf)")
        if not file_path:
            return
            
        self.selected_pdf_path = file_path
        file_name = os.path.basename(file_path)
        file_base_name = os.path.splitext(file_name)[0]
        self.lbl_file_status.setText(f"已選取: {file_name}")
        
        base_doc_id = hashlib.md5(file_base_name.encode('utf-8')).hexdigest()
        
        import chromadb
        db_client = chromadb.PersistentClient(path=CHROMADB_DIR)
        existing_collections = [c.name for c in db_client.list_collections()]
        target_collection_name = f"collection_{base_doc_id}"
        
        self.page_range_box.setVisible(True)
        self.conflict_group_box.setVisible(True)
        self.t1_vlm_config_box.setVisible(True)
        self.btn_run_ocr.setVisible(True)
        self.btn_run_vlm_reconstruct.setVisible(True)
        
        if target_collection_name in existing_collections:
            self.lbl_id_msg.setText("🔎 偵測到重複文件：此檔案過去已建立過快取索引！(直接點擊下方紫色按鈕做 VLM 萃取)")
            self.lbl_id_msg.setStyleSheet("color: #b45309; font-weight: bold;")
            self.radio_keep.setVisible(True)
            self.radio_overwrite.setVisible(True)
            self.radio_keep.setChecked(True)
            self.current_generated_id = base_doc_id
            
            # 💡 核心優化：即使偵測到重複文件，不隱藏綠色按鈕，且讓紫色按鈕保持可用，允許使用者跳過 OCR 直接呼叫 VLM
            self.btn_run_ocr.setVisible(True)
        else:
            self.lbl_id_msg.setText("✨ 偵測到新文件：系統已自動分派識別碼。")
            self.lbl_id_msg.setStyleSheet("color: #1d4ed8; font-weight: bold;")
            self.radio_keep.setVisible(False)
            self.radio_overwrite.setVisible(False)
            self.current_generated_id = base_doc_id
            
        self.ent_doc_id.setText(self.current_generated_id)

    def action_toggle_conflict_strategy(self):
        file_name = os.path.basename(self.selected_pdf_path)
        file_base_name = os.path.splitext(file_name)[0]
        
        if self.radio_keep.isChecked():
            self.current_generated_id = hashlib.md5(file_base_name.encode('utf-8')).hexdigest()
        else:
            unique_seed = f"{file_base_name}_{time.time()}"
            self.current_generated_id = hashlib.md5(unique_seed.encode('utf-8')).hexdigest()
            
        self.ent_doc_id.setText(self.current_generated_id)

    def action_trigger_ocr_pipeline(self):
        self.btn_run_ocr.setEnabled(False)
        self.btn_run_vlm_reconstruct.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_stop_ocr.setEnabled(True)
        self.progress_ocr.setVisible(True)
        
        s_page = self.ent_start_page.text().strip()
        e_page = self.ent_end_page.text().strip()
        start_val = int(s_page) if s_page.isdigit() else None
        end_val = int(e_page) if e_page.isdigit() else None
        
        self.ocr_thread = OcrWorker(
            self.selected_pdf_path, 
            self.current_generated_id, 
            start_page=start_val, 
            end_page=end_val
        )
        self.ocr_thread.finished_signal.connect(self.handle_ocr_success)
        self.ocr_thread.error_signal.connect(self.handle_ocr_error)
        self.ocr_thread.start()

    def action_cancel_ocr(self):
        if self.ocr_thread and self.ocr_thread.isRunning():
            print("[Stop Command] Stopping OCR thread...")
            self.ocr_thread.is_running = False
            self.btn_stop_ocr.setEnabled(False)
        if self.vlm_thread and self.vlm_thread.isRunning():
            print("[Stop Command] Stopping VLM thread...")
            self.vlm_thread.is_running = False
            self.btn_stop_ocr.setEnabled(False)

    def handle_ocr_success(self, total_inserted):
        self.progress_ocr.setVisible(False)
        self.btn_run_ocr.setEnabled(True)
        self.btn_run_vlm_reconstruct.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_stop_ocr.setEnabled(False)
        QMessageBox.information(self, "🎉 成功", f"處理完成！共計變更/寫入 {total_inserted} 個區塊數據。")
        self.refresh_cached_databases()

    def handle_ocr_error(self, error_msg):
        self.progress_ocr.setVisible(False)
        self.btn_run_ocr.setEnabled(True)
        self.btn_run_vlm_reconstruct.setEnabled(True)
        self.btn_select.setEnabled(True)
        self.btn_stop_ocr.setEnabled(False)
        QMessageBox.warning(self, "任務狀態", f"排程已結束: {error_msg}")

    def action_trigger_vlm_reconstruct(self):
        p_val = self.t1_vlm_provider_combo.currentText()
        model_name = self.t1_vlm_model_edit.currentText().strip()
        target_ip = self.t1_vlm_ip_edit.text().strip()
        target_port = self.t1_vlm_port_edit.text().strip()
            
        if not model_name or model_name == "qwen2-vl-7b-instruct" and self.t1_vlm_model_edit.count() == 1:
            reply = QMessageBox.question(
                self, "模型未就緒警告", 
                "未偵測到本地運行的 VLM 模型。是否要嘗試以目前標籤發送請求？(請確保 LM Studio 已按下 Start Server)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
            
        self.btn_run_ocr.setEnabled(False)
        self.btn_run_vlm_reconstruct.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.btn_stop_ocr.setEnabled(True)
        self.progress_ocr.setVisible(True)
        
        self.vlm_thread = VlmReconstructWorker(
            target_doc_id=self.current_generated_id,
            provider=p_val,
            model_name=model_name,
            target_ip=target_ip,
            target_port=target_port
        )
        self.vlm_thread.finished_signal.connect(self.handle_ocr_success)
        self.vlm_thread.error_signal.connect(self.handle_ocr_error)
        self.vlm_thread.start()

    def setup_qa_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        config_grid = QGridLayout()
        config_grid.addWidget(self.create_styled_label("文檔識別碼 (doc_id):"), 0, 0)
        self.ent_qa_doc_id = QLineEdit()
        self.ent_qa_doc_id.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1;")
        config_grid.addWidget(self.ent_qa_doc_id, 0, 1)
        
        config_grid.addWidget(self.create_styled_label("LLM 類別:"), 0, 2)
        self.cbo_provider = QComboBox()
        self.cbo_provider.addItems(["遠端 ollama", "本地 lmstudio", "線上 Groq"])
        self.cbo_provider.setStyleSheet("color: #000000; background-color: #ffffff;")
        self.cbo_provider.currentTextChanged.connect(self.action_refresh_provider_sub_layout)
        config_grid.addWidget(self.cbo_provider, 0, 3)
        layout.addLayout(config_grid)
        
        self.dynamic_param_widget = QWidget()
        self.dynamic_param_layout = QHBoxLayout(self.dynamic_param_widget)
        self.dynamic_param_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.dynamic_param_widget)
        
        self.action_refresh_provider_sub_layout(self.cbo_provider.currentText())
        
        layout.addWidget(self.create_styled_label("請輸入您想查詢的問題:"))
        self.ent_query = QLineEdit()
        self.ent_query.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1;")
        layout.addWidget(self.ent_query)
        
        qa_control_layout = QHBoxLayout()
        self.btn_send_query = QPushButton("🔍 送出檢索並使模型回答")
        self.btn_send_query.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; height: 35px;")
        self.btn_send_query.clicked.connect(self.action_execute_rag_query_pipeline)
        qa_control_layout.addWidget(self.btn_send_query, 4)
        
        self.btn_stop_query = QPushButton("🛑 停止生成")
        self.btn_stop_query.setStyleSheet("background-color: #b91c1c; color: white; font-weight: bold; height: 35px;")
        self.btn_stop_query.setEnabled(False)
        self.btn_stop_query.clicked.connect(self.action_cancel_query)
        qa_control_layout.addWidget(self.btn_stop_query, 1)
        layout.addLayout(qa_control_layout)
        
        self.progress_query = QProgressBar()
        self.progress_query.setRange(0, 0)
        layout.addWidget(self.progress_query)
        self.progress_query.setVisible(False)
        
        panes_layout = QHBoxLayout()
        left_box = QVBoxLayout()
        left_box.addWidget(self.create_styled_label("📋 檢索最終採信之文件脈絡", "color: #0f172a; font-weight: bold;"))
        self.txt_context = QTextEdit()
        self.txt_context.setReadOnly(True)
        self.txt_context.setStyleSheet(f"background-color: #f8fafc; color: #0f172a; font-family: {self.monospace_font_name}; border: 1px solid #e2e8f0;")
        left_box.addWidget(self.txt_context)
        panes_layout.addLayout(left_box)
        
        right_box = QVBoxLayout()
        right_box.addWidget(self.create_styled_label("💡 AI 生成之最佳解答", "color: #166534; font-weight: bold;"))
        self.txt_answer = QTextEdit()
        self.txt_answer.setReadOnly(True)
        self.txt_answer.setStyleSheet("background-color: #f0fdf4; color: #14532d; border: 1px solid #dcfce7;")
        right_box.addWidget(self.txt_answer)
        panes_layout.addLayout(right_box)
        
        layout.addLayout(panes_layout)
        self.tabs.addTab(tab, " 💬 既有文件歷史檢索 Q & A ")

    def action_refresh_provider_sub_layout(self, provider_text):
        while self.dynamic_param_layout.count():
            item = self.dynamic_param_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if provider_text == "ollama":
            self.dynamic_param_layout.addWidget(self.create_styled_label("📡 遠端主機 IP:"))
            self.ent_ol_ip = QLineEdit("")
            self.ent_ol_ip.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1;")
            self.dynamic_param_layout.addWidget(self.ent_ol_ip)
            
            self.dynamic_param_layout.addWidget(self.create_styled_label("Port:"))
            self.ent_ol_port = QLineEdit("11434")
            self.ent_ol_port.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1;")
            self.dynamic_param_layout.addWidget(self.ent_ol_port)
            
            btn_refresh = QPushButton("🔄 刷新模型清單")
            btn_refresh.clicked.connect(self.action_async_fetch_provider_models)
            self.dynamic_param_layout.addWidget(btn_refresh)
            
            self.cbo_models = QComboBox()
            self.cbo_models.setStyleSheet("color: #000000; background-color: #ffffff;")
            self.dynamic_param_layout.addWidget(self.cbo_models)
            
        elif provider_text == "本地 lmstudio":
            btn_refresh = QPushButton("🔄 掃描加載本地 LM Studio")
            btn_refresh.clicked.connect(self.action_async_fetch_provider_models)
            self.dynamic_param_layout.addWidget(btn_refresh)
            
            self.cbo_models = QComboBox()
            self.cbo_models.setStyleSheet("color: #000000; background-color: #ffffff;")
            self.dynamic_param_layout.addWidget(self.cbo_models)
            
        elif provider_text == "線上 Groq":
            self.dynamic_param_layout.addWidget(self.create_styled_label("模型名稱:"))
            self.ent_groq_model = QLineEdit("llama-3.1-8b-instant")
            self.ent_groq_model.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1;")
            self.dynamic_param_layout.addWidget(self.ent_groq_model)
            
            self.dynamic_param_layout.addWidget(self.create_styled_label("API Key:"))
            self.ent_groq_key = QLineEdit()
            self.ent_groq_key.setEchoMode(QLineEdit.EchoMode.Password)
            self.ent_groq_key.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1;")
            self.dynamic_param_layout.addWidget(self.ent_groq_key)

    def action_async_fetch_provider_models(self):
        provider = self.cbo_provider.currentText()
        if provider == "ollama":
            ip = self.ent_ol_ip.text().replace("http://", "").replace("https://", "").strip("/")
            port = self.ent_ol_port.text()
            url = f"http://{ip}:{port}"
            models = get_local_models(url, provider="Ollama")
        #else:
        #    models = get_local_models("http://localhost:1234", provider="LM Studio")
            
        if models:
            self.cbo_models.clear()
            self.cbo_models.addItems(models)
            QMessageBox.information(self, "成功", f"成功載入 {len(models)} 個可用模型。")

    def action_centralize_doc_id_to_qa(self):
        """自動同步目前的識別碼至第二分頁"""
        if self.current_generated_id:
            self.ent_qa_doc_id.setText(self.current_generated_id)

    def action_execute_rag_query_pipeline(self):
        if not self.ent_qa_doc_id.text().strip():
            QMessageBox.warning(self, "提示", "請先輸入有效的文檔識別碼。")
            return
            
        p_val = self.cbo_provider.currentText()
        model_name = ""
        api_key = ""
        target_ip = "localhost"
        target_port = "11434"
        
        if p_val == "線上 Groq":
            model_name = self.ent_groq_model.text()
            api_key = self.ent_groq_key.text()
        else:
            if hasattr(self, 'cbo_models'):
                model_name = self.cbo_models.currentText()
            if p_val == "遠端 ollama":
                target_ip = self.ent_ol_ip.text()
                target_port = self.ent_ol_port.text()

        self.btn_send_query.setEnabled(False)
        self.btn_stop_query.setEnabled(True)
        self.progress_query.setVisible(True)
        
        self.query_thread = QueryWorker(
            user_query=self.ent_query.text(),
            target_id=self.ent_qa_doc_id.text().strip(),
            provider=p_val,
            model_name=model_name,
            api_key=api_key,
            target_ip=target_ip,
            target_port=target_port
        )
        self.query_thread.finished_signal.connect(self.handle_query_success)
        self.query_thread.error_signal.connect(self.handle_query_error)
        self.query_thread.start()

    def action_cancel_query(self):
        if self.query_thread and self.query_thread.isRunning():
            self.query_thread.is_running = False
            self.btn_stop_query.setEnabled(False)

    def handle_query_success(self, context, answer):
        self.txt_context.setPlainText(context)
        self.txt_answer.setPlainText(answer)
        self.progress_query.setVisible(False)
        self.btn_send_query.setEnabled(True)
        self.btn_stop_query.setEnabled(False)

    def handle_query_error(self, error_msg):
        self.txt_context.setPlainText("Process terminated.")
        self.txt_answer.setPlainText(f"Task status: {error_msg}")
        self.progress_query.setVisible(False)
        self.btn_send_query.setEnabled(True)
        self.btn_stop_query.setEnabled(False)

    def setup_inspect_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.create_styled_label("🔐 目標文檔識別碼 (doc_id):"))
        self.ent_inspect_id = QLineEdit()
        self.ent_inspect_id.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1;")
        top_layout.addWidget(self.ent_inspect_id)
        
        btn_load_chunks = QPushButton("🔍 讀取底層 Chunks 列表")
        btn_load_chunks.clicked.connect(self.action_load_database_chunks)
        top_layout.addWidget(btn_load_chunks)
        layout.addLayout(top_layout)
        
        # Add missing txt_doc_id display widget
        display_layout = QHBoxLayout()

        

        display_layout.addWidget(self.create_styled_label("📋 已加載文檔識別碼:"))
        self.txt_doc_id = QLineEdit()
        self.txt_doc_id.setReadOnly(True)
        self.txt_doc_id.setStyleSheet(f"background-color: #e2e8f0; color: #0f172a; font-family: {self.monospace_font_name}; border: 1px solid #cbd5e1;")
        display_layout.addWidget(self.txt_doc_id)
        layout.addLayout(display_layout)
        
        body_layout = QHBoxLayout()

        body_layout.setSpacing(0)  # 讓元件緊密相連
        body_layout.setContentsMargins(0, 0, 0, 0)

        self.list_chunks = QListWidget()
        self.list_chunks.setStyleSheet("background-color: #ffffff; color: #000000; border: 1px solid #cbd5e1;")
        self.list_chunks.currentRowChanged.connect(self.action_display_selected_chunk_text)
        #body_layout.addWidget(self.list_chunks, 1)
        
        # 【關鍵】給左邊元件一個初始固定寬度，這樣右邊的 QTextEdit 就會自動填滿剩下空間
        self.list_chunks.setFixedWidth(250) 
        body_layout.addWidget(self.list_chunks)

        # 2. 插入自訂的拖曳分隔線（綁定控制左邊的 list_chunks）
        self.resize_handle = ResizeHandle(self.list_chunks)
        body_layout.addWidget(self.resize_handle)

        self.txt_chunk_content = QTextEdit()
        self.txt_chunk_content.setReadOnly(True)
        self.txt_chunk_content.setStyleSheet(f"background-color: #f8fafc; color: #000000; font-family: {self.monospace_font_name}; border: 1px solid #cbd5e1;")
        #body_layout.addWidget(self.txt_chunk_content, 2)
        body_layout.addWidget(self.txt_chunk_content)
        
        layout.addLayout(body_layout)
        self.tabs.addTab(tab, " 🔍 資料庫區塊檢視校對器 ")

    def action_load_database_chunks(self):
        target_id = self.ent_inspect_id.text().strip()
        if not target_id:
            QMessageBox.warning(self, "提示", "請輸入 doc_id。")
            return
            
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMADB_DIR)
            collection_name = f"collection_{target_id}"
            
            existing = [c.name for c in client.list_collections()]
            if collection_name not in existing:
                QMessageBox.warning(self, "提示", "資料庫中找不到該識別碼快取。")
                return
                
            collection = client.get_collection(name=collection_name)
            self.cached_inspect_data = collection.get(include=["documents", "metadatas"])
            
            # Populate the txt_doc_id display widget
            self.txt_doc_id.setText(target_id)
            
            self.list_chunks.clear()
            self.txt_chunk_content.clear()
            
            if self.cached_inspect_data and self.cached_inspect_data["ids"]:
                for idx, c_id in enumerate(self.cached_inspect_data["ids"]):
                    meta = self.cached_inspect_data["metadatas"][idx] if self.cached_inspect_data["metadatas"] else {}
                    c_type = meta.get("type", "unknown")
                    p_num = meta.get("page", "?")
                    self.list_chunks.addItem(f"[{idx}] ID: {c_id} (Page: {p_num} | Type: {c_type})")
            else:
                QMessageBox.information(self, "提示", "此文檔中沒有任何區塊數據。")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"讀取 Chunk 發生異常: {e}")

    def action_display_selected_chunk_text(self, row_idx):
        if row_idx < 0 or not hasattr(self, 'cached_inspect_data') or not self.cached_inspect_data:
            return
        try:
            full_text = self.cached_inspect_data["documents"][row_idx]
            self.txt_chunk_content.setPlainText(full_text)
        except Exception:
            self.txt_chunk_content.clear()

    def setup_manager_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        top_bar = QHBoxLayout()
        top_bar.addWidget(self.create_styled_label("📊 系統內既有快取向量資料庫列表：", "color: #0f172a; font-weight: bold;"))
        top_bar.addStretch()
        
        btn_refresh = QPushButton("🔄 重新整理清單")
        btn_refresh.clicked.connect(self.refresh_cached_databases)
        top_bar.addWidget(btn_refresh)
        layout.addLayout(top_bar)
        
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        #self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setHorizontalHeaderLabels(["#", "📄 原始文件名稱", "🎯 頁碼範圍說明", "🔐 歷史文檔識別碼 (doc_id)", "📊 總資料區塊"])
        #self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        self.table.setStyleSheet("""
            QTableWidget { background-color: #ffffff; color: #000000; gridline-color: #cbd5e1; }
            QHeaderView::section { background-color: #e2e8f0; color: #0f172a; font-weight: bold; border: 1px solid #cbd5e1; }
        """)
        layout.addWidget(self.table)
        
        action_bar = QHBoxLayout()
        action_bar.addStretch()
        self.btn_delete_cache = QPushButton("🗑️ 徹底刪除選定之歷史快取")
        self.btn_delete_cache.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 6px 12px;")
        self.btn_delete_cache.clicked.connect(self.action_delete_selected_collection)
        action_bar.addWidget(self.btn_delete_cache)
        layout.addLayout(action_bar)
        
        self.tabs.addTab(tab, " 🗂️ 已快取資料庫管理 ")

    def refresh_cached_databases(self):
        try:
            import chromadb
            client = chromadb.PersistentClient(path=CHROMADB_DIR)
            collections = client.list_collections()
            
            self.table.setRowCount(0)
            for idx, col in enumerate(collections):
                self.table.insertRow(idx)
                
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
                
                self.table.setItem(idx, 0, QTableWidgetItem(str(idx + 1)))
                self.table.setItem(idx, 1, QTableWidgetItem(orig_name))
                self.table.setItem(idx, 2, QTableWidgetItem(page_desc))
                self.table.setItem(idx, 3, QTableWidgetItem(clean_doc_id))
                self.table.setItem(idx, 4, QTableWidgetItem(str(total_chunks)))
        except Exception as e:
            print(f"[UI Warning] Failed to sync collections: {e}")

    def action_delete_selected_collection(self):
        selected_rows = self.table.selectionModel().selectedRows()
        if not selected_rows:
            return
            
        row_idx = selected_rows[0].row()
        doc_id_item = self.table.item(row_idx, 3)
        doc_name_item = self.table.item(row_idx, 1)
        
        if not doc_id_item:
            return
            
        target_doc_id = doc_id_item.text().strip()
        target_doc_name = doc_name_item.text() if doc_name_item else "未知文件"
        
        reply = QMessageBox.question(
            self, "確認刪除", 
            f"您確定要徹底移除《{target_doc_name}》的本地快取資料庫嗎？\n(doc_id: {target_doc_id})",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                import chromadb
                client = chromadb.PersistentClient(path=CHROMADB_DIR)
                client.delete_collection(name=f"collection_{target_doc_id}")
                QMessageBox.information(self, "成功", "該文件快取資料庫已徹底抹除。")
                self.refresh_cached_databases()
            except Exception as e:
                QMessageBox.critical(self, "錯誤", f"刪除失敗: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RagQtApp()
    window.show()
    sys.exit(app.exec())