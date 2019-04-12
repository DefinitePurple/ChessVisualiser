cocoapi/setup.py is an altered file for windows support, original from https://github.com/cocodataset/cocoapi <br />
'extra_compile_args={'gcc': ['/Qstd=c99']},' is the change <br />
make.bat is easier than using a Makefile on windows, Makefile originates from https://github.com/cocodataset/cocoapi <br /> <br />

both config files in training/ originate from samples provided by Tensorflow object detection api and altered by daniel fitzpatrick  <br />
training/labelmap.pbtxt created by daniel fitzpatrick <br />

generate_tfrecord.py originates from https://github.com/datitran/raccoon_dataset and altered by daniel fitzpatrick <br />
xml_to_csv.py originates from https://github.com/datitran/raccoon_dataset <br /> <br />

resizer.py created by daniel fitzpatrick <br /> <br />

train.py originates from Tensorflow Object Detection API <br />
