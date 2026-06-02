# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec — 시각센서 SNN 데모 앱 (onedir, GUI)
from PyInstaller.utils.hooks import collect_all

datas, binaries, hiddenimports = [], [], []
for pkg in ["torch", "torchvision", "snntorch", "scipy", "pandas", "PIL", "openpyxl"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
    except Exception as e:
        print("collect_all skip", pkg, e)

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ['snn_mnist_all'],
    hookspath=[], runtime_hooks=[], excludes=[],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='SensorSNN',
    console=False,          # GUI 앱 (콘솔 창 숨김)
    disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name='SensorSNN')
