import numpy as np
import math
import pandas
import cv2
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Fix error with TF and Keras
import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, ELU, ZeroPadding2D, MaxPooling2D
from keras.layers.convolutional import Convolution2D

batch_size = 128
rows = 32
cols = 64
ch = 3

df = pandas.read_csv('session_data/driving_log.csv', header=0)
df_train, df_test = train_test_split(df, test_size = 0.2, random_state=0)

def crop(img):
    shape = img.shape
    image = img[math.floor(shape[0]/2.5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA)
    return image

def shift_image(image, angle, trange):
    rows,cols,ch = image.shape
    x = (trange/2) - int(100*np.random.uniform())
    angle = angle + (x * .004)
    M = np.float32([[1,0,x],[0,1,0]])
    image = cv2.warpAffine(image,M,(cols,rows))                         
    return image, angle
    
def flip(img, angle):
    # 50% chance of flipping image on vertical axis
    flip = np.random.randint(100)
    if flip < 50:
        img = cv2.flip(img,1)
        angle = -angle
    return img, angle


def process_img_pipeline(feature_image):    
    # crop, change brightness, and convert to RGB
    feature_image = crop(feature_image)
    feature_image = change_brightness(feature_image)
    feature_image = cv2.cvtColor(feature_image, cv2.COLOR_BGR2RGB)
    return feature_image

def normalize_data_pipeline(feature_image, angle):
    
    # decide if we'll flip it 
    feature_image, angle = flip(feature_image, angle)
    
    # shift the image randomly and adjust steering angle by .004 per pix
    # simulates new positions on the track that the car never actually took
    #feature_image, angle = shift_image(feature_image, angle, 100)
    
    return feature_image, angle
    
def batch_data_generator(X_train, y_train, batch_size):
    bx = np.zeros((batch_size, rows, cols, ch), dtype=np.uint8)
    by = np.zeros(batch_size, dtype=np.float)
    i = 0
    
    while 1:
        
        # randomly choose left, center, or right image
        idx = np.random.randint(len(y_train))
        angle = y_train.iloc[idx]
        rand = np.random.randint(3)
        
        if rand == 0:
            head, fname = os.path.split(X_train.left.iloc[idx])
            img = cv2.imread('./session_data/IMG/' + fname)
            angle += .15
        elif rand == 1:
            head, fname = os.path.split(X_train.center.iloc[idx])
            img = cv2.imread('./session_data/IMG/' + fname)
        else:
            head, fname = os.path.split(X_train.right.iloc[idx])
            img = cv2.imread('./session_data/IMG/' + fname)
            angle -= .15
            
        # new udacity sim only recorded center
        if img is not None:
            # normalize/augment our data. flip, shift, etc.
            img, angle = normalize_data_pipeline(img, angle)
            img = process_img_pipeline(img)
            bx[i] = img
            by[i] = angle
            i += 1
        else:
            pass
         
        if i >= batch_size-1:
            yield bx, by
            # reset
            bx = np.zeros((batch_size, rows, cols, ch), dtype=np.uint8)
            by = np.zeros(batch_size, dtype=np.float)
            i = 0

# a CNN similar to comma.ai model
def get_model(time_len=1):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0,
            input_shape=(rows, cols, ch),
            output_shape=(rows, cols, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(128, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Dropout(.5))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(1))
    
    model.compile(optimizer="adam", loss="mse", lr=0.0001)
    return model

model = get_model()
n_samples_per_epoch = len(df_train.steering)
train_gen = batch_data_generator(df_train, df_train.steering, batch_size)
valid_gen = batch_data_generator(df_test, df_test.steering, batch_size)
history = model.fit_generator(train_gen,
                                  validation_data=valid_gen,
                                  samples_per_epoch=n_samples_per_epoch,
                                  nb_val_samples=n_samples_per_epoch,
                                  nb_epoch=9)
print("training complete!")

# change to true if you want to save new model
if False:
  json = model.to_json()
  model.save_weights("../model.h5")
  with open("../model.json", mode="w+") as f:
      f.write(json)
      f.close()
  print("model saved")
