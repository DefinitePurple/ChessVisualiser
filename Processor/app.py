from random import shuffle
from keras.optimizers import RMSprop
from keras.models import Sequential, model_from_json
from keras.layers import *
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import glob
import h5py
import numpy as np
import argparse
from collections import Counter

CATEGORIES = ['Black', 'White']
IMAGES_DIR = "data/images/"
DATA_H5_PATH = 'data/data.h5'
MODEL_H5_PATH = 'data/models/model.h5'
MODEL_JSON_PATH = 'data/models/model.json'
WEIGHTS_PATH = 'data/models/weight.h5'
EPOCHS = 10
BATCH_SIZE = 32
IMG_SIZE = 101


def createModel():
    model = Sequential()
    model.add(Dense(IMG_SIZE * IMG_SIZE, input_shape=(IMG_SIZE, IMG_SIZE, 3)))

    model.add(Conv2D(IMG_SIZE * IMG_SIZE * 3, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(IMG_SIZE * IMG_SIZE * 3, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def createDataset():
    white = glob.glob(IMAGES_DIR + 'white/*.jpg')  # 0
    black = glob.glob(IMAGES_DIR + 'black/*.jpg')  # 1

    both = white + black
    labels = [[0, 1] if 'black' in path else [1, 0] for path in both]

    zipped = list(zip(both, labels))
    shuffle(zipped)
    train_images, train_labels = zip(*zipped)
    train_shape = (len(train_images), IMG_SIZE, IMG_SIZE, 3)

    f = h5py.File(DATA_H5_PATH, mode='w')
    f.create_dataset("train_img", train_shape, np.int8)
    f.create_dataset("train_labels", (len(train_images), 2), np.int8)
    f["train_labels"][...] = train_labels

    # a numpy array to save the mean of the images
    mean = np.zeros(train_shape[1:], np.float32)
    # loop over train addresses
    for i in range(len(train_images)):
        addr = train_images[i]
        img = cv2.imread(addr)
        img = cv2.resize(img, (101, 101), interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        f["train_img"][i, ...] = img[None]

    f.close()


def train(model, X, Y, epochs=50):
    model.fit(X, Y, epochs=epochs, batch_size=32, shuffle="batch")
    return model


def save_model(model):
    model.save(MODEL_H5_PATH)
    with open(MODEL_JSON_PATH, "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(WEIGHTS_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Virtualise some chess')
    parser.add_argument('--train', '-t', type=int, help='Retrain model - include number of epochs -t 50')
    parser.add_argument('--dataset', '-d', action='store_true', help='Recreate dataset')
    parser.add_argument('--model', '-m', action='store_true', help='Recreate model')
    parser.add_argument('--predict', '-p', type=str, help='Path to input for prediction')

    args = parser.parse_args()

    if args.dataset is True:
        print('Recreating Dataset')
        createDataset()

    if args.model is False:
        model = model_from_json(open(MODEL_JSON_PATH, 'r').read())
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    else:
        print('Recreating Model')
        model = createModel()



    if args.train is not None and args.train is not 0:
        h5f = h5py.File(DATA_H5_PATH, 'r', driver='core')
        X, Y = h5f['train_img'], h5f['train_labels']
        X = X[()].reshape([-1, IMG_SIZE, IMG_SIZE, 3])
        Y = Y[()].reshape([-1, 2])

        model.fit(X, Y, epochs=args.train, batch_size=32, shuffle="batch")
        model.save(MODEL_H5_PATH)
        with open(MODEL_JSON_PATH, "w") as json_file:
            json_file.write(model.to_json())
        model.save_weights(WEIGHTS_PATH)

    #  Do Predictions
    if args.predict is not None:
        print('Using image at {}'.format(args.predict))
        source = cv2.imread(args.predict)
        X = cv2.resize(source, (101, 101), interpolation=cv2.INTER_CUBIC)
        X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)
        X = X[()].reshape([-1, IMG_SIZE, IMG_SIZE, 3])

        predictions = []
        prediction = model.predict(X)

        print(np.argmax(prediction[0]))
        if np.argmax(prediction[0]) == 0:
            print('The piece is Black')
        elif np.argmax(prediction[0]) == 1:
            print('The piece is white')
        plt.imshow(source)
        plt.show()
