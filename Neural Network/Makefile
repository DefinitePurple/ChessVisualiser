all:
    # install pycocotools locally
	python setup.py build_ext --inplace
	rd /s /q build

install:
	# install pycocotools to the Python site-packages
	python setup.py build_ext install
	rm -rf build