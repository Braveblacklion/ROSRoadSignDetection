# L. Rozmarynowski; Sentdex(Youtube-Tutorial)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time


gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.60)  # specify fraction of gpu for model
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))   # load specification in Session
# multiple Sessions possible (0.333 * 3 = 100% --> max 3 Sessions parallel possible)

X = pickle.load(open("X3.pickle", "rb"))                 # import Dataset
y = pickle.load(open("y3.pickle", "rb"))                 # import labels

X = tf.keras.utils.normalize(X, axis=1)                 # Normalize Dataset instead of 0 - 255 from 0.000 - 1.000

# dense_layers = [0, 2, 3]
# layer_sizes = [64, 128]
# conv_layers = [3]

dense_layers = [2]
layer_sizes = [32]
conv_layers = [3]
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            tensorboard = TensorBoard(log_dir='logs2\{}'.format(NAME))  # create Tensorboard objekt, Files
            print(NAME)

            model = Sequential()                                    # create Model

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))  # first conv Layer, 64 Nodes, input Dataset
            model.add(Activation('relu'))                           # Activate Conv Layer
            model.add(MaxPooling2D(pool_size=(2, 2)))               # first pooling Layer

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))                           # conv Layer
                model.add(Activation('relu'))                           # Activate Conv Layer
                model.add(MaxPooling2D(pool_size=(2, 2)))               # pooling Layer

            model.add(Flatten())                                    # 2D image to 1D images for Dense Layers

            for l in range(dense_layer):
                model.add(Dense(layer_size*2))
                model.add(Activation('relu'))
                model.add(Dropout(0.2))

            model.add(Dense(43))                                    # Output Layer, Sum of Output options (Sum Signs)
            model.add(Activation('softmax'))                        # Activation output Layer

            # a few more layers may be necessary

            model.compile(loss='sparse_categorical_crossentropy',          # how to calculate error
                          optimizer='adam',                         # there are more, adam is default optimizer
                          metrics=['accuracy'])                     # what to track

            model.fit(X, y, batch_size=64,                          # input, pictures parallel processed (20-200), depends Sum(pic)
                      validation_split=0.2,                         # %test_data split for val_data 0.2 = 20% (max 30%)
                      epochs=350,                                     # learning cycles
                      callbacks=[tensorboard])                      # create Callback for tensorboard
            model.save(r"C:\**\ki_s\{}".format(NAME))

