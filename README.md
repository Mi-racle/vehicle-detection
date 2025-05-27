环境配置：

1. pip install -r requirements.txt


打包方法：

1. pip install pyinstaller -i https://pypi.tuna.tsinghua.edu.cn/simple
2. pyinstaller.exe -D -w -n CV-AI run.py -i icon.ico --collect-submodules=numpy.f2py --hiddenimport=scipy._lib.array_api_compat.numpy.fft --hiddenimport=scipy.special._special_ufuncs --add-data configs;configs --add-data ui/assets;ui/assets --add-data ocr/configs;ocr/configs --add-data ocr/dicts;ocr/dicts --noconfirm
