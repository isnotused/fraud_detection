# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

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
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='App',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
