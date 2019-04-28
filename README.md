# Chess Visualiser
## Demo
https://youtu.be/UibvMQHDLIA

## Windows Support Only - Use other operating systems at your own risk

## Install & running instructions for training the neural network
``` bash
Clone this repo anywhere 'git clone https://github.com/DefinitePurple/ChessVisualiser.git'
Clone COCOAPI to same directory 'git clone https://github.com/cocodataset/cocoapi'

Install Anaconda Python 3.7 for Windows - https://www.anaconda.com/distribution/
Once installed run 'conda create -n chess pip python=3.6'
To activate the environment 'conda activate chess'

Install tensorflow GPU 'pip install --ignore-installed --upgrade tensorflow-gpu==1.12.0'
Install rest of modules 'pip install -r requirements.txt'
If you encounter a missing module 'pip install [module name]'

Make a directory called tensorflow in C: drive 'C:\tensorflow'
Clone Tensorflow Object Detection API to C:\chess 'git clone https://github.com/tensorflow/models.git'
set PYTHONPATH=C:\tensorflow\models;C:\tensorflow\models\research;C:\tensorflow\models\research\slim
The above command will have to be run everytime you run 'conda activate'

Copy make.bat and setup.py from this repo Neural Network/cocoapi into cocoapi/PythonApi
run make.bat in cocoapi/PythonApi
cp -r cocoapi\PythonApi\pycocotools C:\tensorflow\models\research

cd to C:tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.
python setup.py build
python setup.py install

copy xml_to_csv.py in Neural Network to C:\tensorflow\models\research\object_detection
copy generate_tfrecord.py in Neural Network to C:\tensorflow\models\research\object_detection
copy train.py in Neural Network to C:\tensorflow\models\research\object_detection
copy training\ in Neural Network to C:\tensorflow\models\research\object_detection
copy images\ from Data to C:\tensorflow\models\research\object_detection

In C:\tensorflow\models\research\object_detection\training\faster_rcnn_resnet101_coco.config change all instances of paths to the relative paths on your pc
Example: Change D:/ChessVisualiser/models/research/object_detection/training/labelmap.pbtxt to C:/tensorflow/models/research/object_detection/training/labelmap.pbtxt

From C:\tensorflow\models\research\object_detection
python xml_to_csv.py
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_resnet101_coco.config

To view graphs run 'tensorboard logdir=training/'

When training is complete run following command, where XXXX is the ckpt number in training/ 
'python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph'
frozen_inference_graph.pb in C:/tensorflow/models/research/object_detection/inference_graph is the file used with detection
```

## Install & running instructions for webserver
``` bash
Clone this repo anywhere 'git clone https://github.com/DefinitePurple/ChessVisualiser.git'
Clone COCOAPI to same directory 'git clone https://github.com/cocodataset/cocoapi'

Install Anaconda Python 3.7 for Windows - https://www.anaconda.com/distribution/
Once installed run 'conda create -n chess pip python=3.6'
To activate the environment 'conda activate chess'

Install tensorflow GPU 'pip install --ignore-installed --upgrade tensorflow-gpu==1.12.0'
Install rest of modules 'pip install -r requirements.txt'
If you encounter a missing module 'pip install [module name]'

To run the server, cd into this repos root and run 'run.bat'
```

## Installing & running instructions for Demo
``` bash
Clone this repo anywhere 'git clone https://github.com/DefinitePurple/ChessVisualiser.git'
Clone COCOAPI to same directory 'git clone https://github.com/cocodataset/cocoapi'

Install Anaconda Python 3.7 for Windows - https://www.anaconda.com/distribution/
Once installed run 'conda create -n chess pip python=3.6'
To activate the environment 'conda activate chess'

Install tensorflow GPU 'pip install --ignore-installed --upgrade tensorflow-gpu==1.12.0'
Install rest of modules 'pip install -r requirements.txt'
If you encounter a missing module 'pip install [module name]'

cd into Demo
python multiImageDemo.py
```

## File Ownership
Each directory has their own README with an explaination of file ownership <br />
ChessVisualiser/README.md <br />
Data/README.md <br />
Demo/README.md <br />
Neural Network/README.md <br />
run.bat created by daniel fitzpatrick <br />
