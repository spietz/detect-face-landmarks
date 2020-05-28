import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
from matplotlib import cm
plt.ion()

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')

# blog post on face landmark detection:
# https://towardsdatascience.com/face-landmark-detection-with-cnns-tensorflow-cf4d191d2f0

# blog post on transfer learning:
# https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/

# face images with marked landmarks dataset from:
# https://www.kaggle.com/drgilermo/face-images-with-marked-landmark-points/


# PARAMETERS ###

num_epochs = 10
batch_size = 32
learning_rate = 1e-4


# DATA ###

IMAGE_SIZE = 96
COLOR_CHANNELS = 1          # datasets consists of bw images
NUM_LANDMARKS = 30          # this is acutally number of landmarks times dimensionality

landmarks = pd.read_csv( "dataset/facial_keypoints.csv")  # load csv
# not all images contain all landmarks
# -> should not be a problem, but for now keep only images with all landmarks
num_missing_landmarks = landmarks.isnull().sum( axis=1 )
all_landmarks_present_ids = np.array(num_missing_landmarks == 0)

print("num (images, landmarks*2)", landmarks.shape)
print(landmarks.columns)
print("images where all landmarks has been marked:", sum(all_landmarks_present_ids))

d = np.load( "dataset/face_images.npz")
#print("face_images.npz contains:")
# for k in d.keys():
#     print(k, type(d[k]))
dataset = d[ 'face_images' ].T
dataset = np.reshape(dataset, (-1, IMAGE_SIZE, IMAGE_SIZE))  # grayscale -> 1 component

# we need color so we use a color lookup table (intensity 0-1 -> (r,g,b,alpah) (0-1,0-1,0-1,0-1))
lut = cm.Greys_r

numSamples = len(np.nonzero(all_landmarks_present_ids))
# images: [numSamples x height x width x channels]
images_bw = dataset[all_landmarks_present_ids, :, :]
images_bw = images_bw / 255  # normalize
images = lut(images_bw)[:,:,:,:3]  # 0-1, rbg skip alpha
# landmarks: [numSamples x num_landmarks]
landmarks = landmarks.iloc[all_landmarks_present_ids, :].reset_index( drop=True ).values
landmarks = landmarks / IMAGE_SIZE

# divide into test, train sets
x_train, x_test, \
    y_train, y_test = train_test_split(images , landmarks , test_size=0.3 )


# MODEL ###

# we use a pre-trained model as basis for the landmark detector
# we do not load fully connected output layers at the end
# we specify the shape of our input data (VGG16 is made for 224x224x3)
base_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))  # must be 3 channel
base_model = VGG16(input_tensor=base_input, include_top=False)
 # mark loaded layers as not trainable, note how summary changes
for layer in base_model.layers:
	layer.trainable = False

# add layers, flatten->dense->output
flat1 = Flatten()(base_model.output)
dense1 = Dense(1024, activation='relu')(flat1)
output = Dense(NUM_LANDMARKS, activation=None)(dense1)
# define the new model
model = Model(inputs=base_input, outputs=output)

model.summary()


# TRAINING ###

# define loss metric as mean squared error because this is a regression problem
model.compile( loss=tf.keras.losses.mean_squared_error,
               optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
               metrics=['mse'] )

# do the training
model.fit(x=x_train, y=y_train, epochs=num_epochs, batch_size=batch_size)


# TESTING ### 

preds = model.evaluate(x = x_test, y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
print ("[mean distance in \% off image size] = " + str(np.sqrt(preds[1])*100.0))


# PLOTS ###

# training samples
fig1 = plt.figure(1)
fig1.clf()

rand_idx = np.random.choice(x_train.shape[0], size=4)

y_pred = model.predict( x_train[rand_idx,:,:,:] )

for i, idx in enumerate(rand_idx):

    img = x_train[idx,:,:,:]

    ax = fig1.add_subplot(2,2,i+1)

    lm_x_true = y_train[idx, 0::2]
    lm_y_true = y_train[idx, 1::2]
    ax.imshow(np.transpose(img, axes=[1,0,2]))
    ax.plot(lm_x_true*IMAGE_SIZE, lm_y_true*IMAGE_SIZE, 'gx')

    lm_x_pred = y_pred[i, 0::2]
    lm_y_pred = y_pred[i, 1::2]
    ax.plot(lm_x_pred*IMAGE_SIZE, lm_y_pred*IMAGE_SIZE, 'rx')

    # print( np.mean((y_pred - y_train[rand_idx, :])**2) )

# test samples
fig2 = plt.figure(2)
fig2.clf()

rand_idx = np.random.choice(x_test.shape[0], size=4)

y_pred = model.predict( x_test[rand_idx,:,:,:] )

for i, idx in enumerate(rand_idx):

    img = x_test[idx,:,:,:]

    ax = fig2.add_subplot(2,2,i+1)
    
    lm_x_true = y_test[idx, 0::2]
    lm_y_true = y_test[idx, 1::2]
    ax.imshow(np.transpose(img, axes=[1,0,2]))
    ax.plot(lm_x_true*IMAGE_SIZE, lm_y_true*IMAGE_SIZE, 'gx')

    
    lm_x_pred = y_pred[i, 0::2]
    lm_y_pred = y_pred[i, 1::2]
    ax.plot(lm_x_pred*IMAGE_SIZE, lm_y_pred*IMAGE_SIZE, 'rx')

    # print( np.mean((y_pred - y_train[rand_idx, :])**2) )
    
    
