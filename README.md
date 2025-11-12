# 打包
在项目环境中使用 PyInstaller 打包 Streamlit 应用。确保你已经安装了 PyInstaller：
```BASH
pyinstaller launcher.py ^
    --name "App" ^                             
    --onefile ^                                     
    --windowed ^                                   
    --add-data "App;App" ^ 
    --collect-all "streamlit" ^                     
    --clean                                         
```
说明：
pyinstaller launcher.py ^
    --name "App" ^                                  # 生成的exe文件名
    --onefile ^                                     # 打包成一个单一的exe文件
    --windowed ^                                    # 隐藏控制台窗口 (对于streamlit应用通常不需要控制台)
    --add-data "my_streamlit_app;my_streamlit_app" ^ # 将整个 Streamlit 目录添加到打包中
    --collect-all "streamlit" ^                     # 确保收集所有 Streamlit 相关的隐藏导入
    --clean                                         # 清理临时文件

注意：
^ 是 Windows CMD 的续行符，但在 PowerShell 里应该用反引号 ` 或直接写成一行
```PowerShell
# 安装依赖（任选其一）
pip install -r requirements.txt
# 或 uv:
uv pip install

# 分行写法（PowerShell 用反引号`）
pyinstaller launcher.py `
    --name "App" `
    --onefile `
    --windowed `
    --add-data "APP;APP" `
    --collect-all "streamlit" `
    --clean

# 一行写法
pyinstaller launcher.py --name "App" --onefile --windowed --add-data "APP;APP" --collect-all streamlit --clean
```
打包完成后，exe 文件会在项目目录下的 dist 文件夹中:
`c:当前目录\dist\App.exe`

lunix
```BASH
pyinstaller launcher.py \
    --name "App" \
    --onefile \
    --windowed \
    --add-data "APP:APP" \
    --collect-all streamlit \
    --hidden-import networkx \
    --hidden-import scipy \
    --clean
```


# 解决 streamlit_option_menu 资源文件缺失问题
在APP.spec文件中，添加datas参数，和 hiddenimports 指定streamlit_option_menu的资源文件路径：

datas = [
    ('APP', 'APP'),
    ('C:/Users/Administrator/Desktop/专利书写2025/202511/基于大数据的金融领域消费欺诈检测方法/【研发证明材料】基于大数据的金融领域消费欺诈检测方法/fraud_detection/.venv/Lib/site-packages/streamlit_option_menu/frontend/dist', 'streamlit_option_menu/frontend/dist')
]
binaries = []
hiddenimports = [
    'networkx', 
    'pandas', 
    'scikit-learn', 
    'datetime', 
    'matplotlib', 
    'numpy', 
    'plotly',
    'scipy',
    'scipy.signal',
    'scipy.optimize',
    'scipy.stats',
    'scipy.special',
    'scipy.linalg',
    'scipy.integrate',
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner',
    'streamlit_option_menu',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.preprocessing',
    'sklearn.model_selection',
    'sklearn.metrics',
    ]




# 表情符号
👥 