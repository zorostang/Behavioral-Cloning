import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from threading import Timer
import cv2
import matplotlib.image as mpimg
import math
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

class Car():
    def __init__(self):
        self.throttle=0.2
        self.recovering = False
        
    def throttle_decision(self, steering_angle):
        # if the steering angle is high, then we should slow down and recover
        # if it's low and we've taken enough time for recovery..speed back up!
        if not self.recovering:
            if abs(steering_angle) < 0.05:
                self.throttle = 0.2
            elif abs(steering_angle) < 0.15 and abs(steering_angle) > 0.05:
                self.throttle = 0.2
            else:
                self.throttle = 0.1
                self.recover().start()
        return self.throttle
    
    def recover(self):
        print("Recovering...")
        t = Timer(4, self.stop_recovery)
        self.recovering = True
        return t
    def stop_recovery(self):
        self.recovering = False
    
car = Car()
@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    
    # The current throttle of the car
    throttle = data["throttle"]
    
    # The current speed of the car
    speed = data["speed"]
    
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    
    # process the image NOTE: img_to_array reads image as RGB
    image_array = img_to_array(image).astype('uint8')
    processed_image = process_image(image_array)
    transformed_image_array = processed_image[None, :, :, :]
    
    # This model currently assumes that the features of the model are just the images. Feel free to change this.
    steering_angle = float(model.predict(transformed_image_array, batch_size=1))
    
    # The driving model
    throttle = car.throttle_decision(steering_angle)
    print(steering_angle, throttle)
    send_control(steering_angle, throttle)

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
    'steering_angle': steering_angle.__str__(),
    'throttle': throttle.__str__()
    }, skip_sid=True)
    
def crop(img):
    shape = img.shape
    image = img[math.floor(shape[0]/2.5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image, (32, 16), interpolation=cv2.INTER_AREA)
    return image

def change_color_space(img, cspace="RGB"):
    # convert image to new color space (if specified)
    # image must be in RGB format, not BGR 
    if cspace != 'RGB':
        if cspace == 'HSV':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif cspace == 'LUV':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif cspace == 'HLS':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif cspace == 'YUV':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif cspace == 'YCrCb':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        elif cspace == 'BGR':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif cspace == 'Gray':
            image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else: image = np.copy(img)
    return image

def process_image(img):
    img = change_color_space(img, cspace="RGB")
    img = crop(img)
    #img = change_brightness(img)
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        #   model = model_from_json(json.loads(jfile.read()))\
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)