# Chess Visualiser
The webserver prototype was created under the guidance of Flask's own tutorial
http://flask.pocoo.org/docs/1.0/tutorial/

## Demo
`New demo Video coming soon`

## Operating system
Windows Support Only <br />
Use other distros at your own risk <br />

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


## How to use
`git clone https://github.com/DefinitePurple/ChessVisualiser.git`<br />
`cd ChessVisualiser`
### Webserver
__Linux__<br />
`./run.sh`<br />
__Windows__<br />
`run.bat`<br />
### Neural Network
__NOTE: I use the windows python launcher 'py' to run python as its easier for running multiple versions of python.__ https://docs.python.org/3/using/windows.html#launcher
<br />
__Create model - Close all other applications. CPU Intensive__<br />
`py -3 app.py -m`<br />
__Train model - Due to small dataset, lower number of epochs recommended__<br />
`py -3 app.py -t [Number of epochs]`<br />
`py -3 app.py -t 3`<br />
__Predict image__<br />
`py -3 app.py -p [Path to file]`<br />
`py -3 app.py -p ./test/test1.jpg`<br />

## File Ownership - ([Path to file] - [Owner])
run.bat - Daniel Fitzpatrick <br/>
run.sh - Daniel Fitzpatrick <br/>
ChessVisualiser/_\_init__.py - Daniel Fitzpatrick <br/>
ChessVisualiser/auth.py - Daniel Fitzpatrick <br/>
ChessVisualiser/db.py  - Daniel Fitzpatrick <br/>
ChessVisualiser/match.py  - Daniel Fitzpatrick <br/>
ChessVisualiser/schema.sql  - Daniel Fitzpatrick <br/>
ChessVisualiser/site.py  - Daniel Fitzpatrick <br/>
ChessVisualiser/static/css/chessboard.css - http://chessboardjs.com/ <br/>
ChessVisualiser/static/css/chessboard.min.css - http://chessboardjs.com/ <br/>
ChessVisualiser/static/css/style.css - Daniel Fitzpatrick <br/>
ChessVisualiser/static/js/base.js - Daniel Fitzpatrick <br/>
ChessVisualiser/static/js/chessboard.js - http://chessboardjs.com/ <br/> 
ChessVisualiser/static/js/chessboard.min.js - http://chessboardjs.com/ <br/>
ChessVisualiser/static/js/jsquery.js - https://jquery.com/ <br/>
ChessVisualiser/static/js/jquery.min.js - https://jquery.com/ <br/>
ChessVisualiser/static/js/match.js - Daniel Fitzpatrick <br/>
ChessVisualiser/static/js/matches.js - Daniel Fitzpatrick <br/>
ChessVisualiser/templates/ - All files created by Daniel Fitzpatrick <br/>
Processor/app.py - Daniel Fitzpatrick <br />
Processor/data/data.h5 - generated by the code <br />
Processor/data/models/ - all files generated by code <br />
Processor/data/images/ - All images created by Daniel Fitzpatrick
Processor/test/ - All images created by Daniel Fitzpatrick
