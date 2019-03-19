move setup.py and make.bat to cocoapi/PythonAPI/ when installing the coco api
then run make.bat

setup.py is an altered file for windows support
'extra_compile_args={'gcc': ['/Qstd=c99']},' is the change
make.bat is an easier than using a Makefile on windows
