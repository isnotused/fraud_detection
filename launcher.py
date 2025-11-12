import streamlit.web.cli as stcli
import sys
import os

# 将 Streamlit 应用的主目录添加到 PYTHONPATH，以便Python能找到相对路径的模块
# 假设你的launcher.py和my_streamlit_app在同一个父目录
# 或者你可以在打包时使用 --add-data 参数，但我发现这种方法更可靠
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设 streamlit_app 目录与 launcher.py 在同一级
app_root_dir = os.path.join(current_dir, 'App') # 根据你的实际目录名修改
sys.path.insert(0, app_root_dir)


if __name__ == "__main__":
    # 模拟命令行参数：streamlit run App/head.py
    # 注意：这里直接指定了完整的路径
    sys.argv = [
        "streamlit", "run",
        os.path.join(app_root_dir, "head.py"), # 确切的主应用入口文件
        "--global.developmentMode", "false", # 生产模式，禁用热重载
        "--server.headless", "false",       # 确保打开浏览器
        "--browser.gatherUsageStats", "false" # 禁用使用统计
        # 你可以添加其他 Streamlit 命令行参数, 例如 --server.port 8501
    ]
    stcli.main()

