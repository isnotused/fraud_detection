import streamlit.web.cli as stcli
import sys
import os

# 统一资源定位：开发环境与 PyInstaller 冻结环境
def get_base_dir() -> str:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS  # PyInstaller 解包的临时目录
    return os.path.dirname(os.path.abspath(__file__))

base_dir = get_base_dir()

# 统一使用实际目录名：APP（注意大小写）
app_root_dir = os.path.join(base_dir, 'APP')

# 将 APP 目录加入模块搜索路径
if app_root_dir not in sys.path:
    sys.path.insert(0, app_root_dir)

# 运行时切换到 APP 目录，保证相对路径（如 static/.streamlit）可用
try:
    os.chdir(app_root_dir)
except Exception:
    pass

# 常用环境变量优化 Streamlit 行为（可按需调整）
os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')
os.environ.setdefault('STREAMLIT_GLOBAL_DEVELOPMENT_MODE', 'false')

if __name__ == "__main__":
    # 以绝对路径启动 Streamlit 应用
    head_path = os.path.join(app_root_dir, "head.py")
    sys.argv = [
        "streamlit", "run", head_path,
        "--server.headless", "false",  # 需要自动打开浏览器可设为 false
        "--browser.gatherUsageStats", "false"
        # 可追加："--server.port", "8501"
    ]
    stcli.main()

