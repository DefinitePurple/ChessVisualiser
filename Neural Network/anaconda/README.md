Follow the installation steps on the object detection api first
    Note when installing cocoapi follow the steps in this repositories cocoapi directory for windows installation
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

conda create -n tensorflow1 pip python=3.6

activate tensorflow1
pip install --ignore-installed --upgrade tensorflow-gpu==1.12.0

conda install -c anaconda protobuf
pip install pillow
pip install lxml
pip install Cython
pip install jupyter
pip install matplotlib
pip install pandas
pip install opencv-python

set PYTHONPATH=D:\tensorflow1\models;D:\tensorflow1\models\research;D:\tensorflow1\models\research\slim

cd D:\tensorflow1\models\research
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
python setup.py build
python setup.py install

cd object_detection
jupyter notebook object_detection_tutorial.ipynb

