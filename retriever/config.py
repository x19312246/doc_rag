"""" 設定chromadb """
import os
import sys

# 💡 支援 pyinstaller 打包(但傾向不打包，整檔太大)
if getattr(sys, 'frozen', False):
    # 在 PyInstaller exe 环境中，sys._MEIPASS 指向临时解压目录
    BASE_PATH = sys._MEIPASS
else:
    # 开发环境：基于 config.py 所在位置推导
    BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHROMADB_DIR = os.path.join(BASE_PATH, "chromadb_storage")

# 确保目录存在
os.makedirs(CHROMADB_DIR, exist_ok=True)