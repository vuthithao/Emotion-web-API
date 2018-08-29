# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py
from __future__ import division, print_function
import io


# coding=utf-8
import sys
import os
import glob
import re
import scipy
import io
from PIL import Image
import base64
import tensorflow as tf
import align.detect_face



# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json

# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import numpy as np
import cv2
from keras.preprocessing import image
import time

# initialize our Flask application and the Keras model
app = Flask(__name__)

# facenet 
image_size=160
margin= 44
gpu_memory_fraction=1.0

def load_and_align_data(img, image_size,margin, gpu_memory_fraction):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    #img = scipy.misc.imread(os.path.expanduser(image_path))
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    if (len(bounding_boxes)==0):
        bb=0
        have_face = 0
    else:
        det = np.squeeze(bounding_boxes[0,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2 - bb[0], img_size[1])
        bb[3] = np.minimum(det[3]+margin/2 - bb[1], img_size[0])
        have_face = 1
    return bb,have_face


model = model_from_json(open("/home/thaovu/simple-keras-rest-api/models/model_4layer_2_2_pool.json", "r").read())
model.load_weights('/home/thaovu/simple-keras-rest-api/models/model_4layer_2_2_pool.h5') #load weights


def model_predict(img, model):
#facenet 
    #img = cv2.imread(img_path)
    detect_face, have_face= load_and_align_data(img,image_size,margin,gpu_memory_fraction)
    preds = []
    detect = []
    if (have_face!=0):
        detect_face = np.reshape(detect_face,(-1,4)) 
        
        for (x,y,w,h) in detect_face:
            detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY) #transform to gray scale
            detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48

            img_pixels = image.img_to_array(detected_face)
            img_pixels = np.expand_dims(img_pixels, axis = 0)

            img_pixels /= 255 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]

            #------------------------------

            predictions = model.predict(img_pixels) #store probabilities of 7 expressions
            preds.append(predictions)
            detect.append(detect_face)
    return preds,have_face, detect


@app.route('/predict', methods = ['POST'])
def predict():
    data = {"success": False}
    if request.method == "POST":
        # Get the file from post request
        f = request.form["image"]
        f = f.replace("\\n","\n")
        f = str.encode(f)
        img= base64.decodestring(f)
        print(type(img))
        img = Image.open(io.BytesIO(img))
        print(type(img))
        img = np.array(img)
        print(img.shape)

        # Make prediction
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        data["position"] = []
        data["result"] = []
        data["predictions"] = []
        preds, have_face, detect_face = model_predict(img, model)
        if (have_face != 0):
            for j in range(len(preds)):
                data["position"] = [{"x": float(detect_face[j][0][0]), "y": float(detect_face[j][0][1]),
                                     "w": float(detect_face[j][0][2]), "h": float(detect_face[j][0][3])}]
                max_index = np.argmax(preds[j][0])
                data["result"] = [{"result": emotions[max_index]}]

                for i in range(len(emotions)):
                    r = {"label": emotions[i], "probability": round(preds[j][0][i] * 100, 2)}
                    data["predictions"].append(r)
        data["success"] = True
    return jsonify(data)
    # return render_template('upload.html')
    #     if request.files.get("image"):
    #     # Get the file from post request
    #         f = request.files["image"]
    #     # Save the file to ./uploads
    #         basepath = os.path.dirname(__file__)
    #         file_path = os.path.join(
    #     basepath,'uploads', secure_filename(f.filename))
    #         f.save(file_path)
    #
    #     # Make prediction
    #         emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
    #         data["position"] = []
    #         data["result"] = []
    #         data["predictions"] = []
    #         preds, have_face, detect_face = model_predict(file_path, model)
    #         if (have_face != 0):
    #             for j in range(len(preds)):
    #                 data["position"] = [{"x": float(detect_face[j][0][0]), "y": float(detect_face[j][0][1]),
    #                                      "w": float(detect_face[j][0][2]), "h": float(detect_face[j][0][3])}]
    #                 max_index = np.argmax(preds[j][0])
    #                 data["result"] = [{"result": emotions[max_index]}]
    #
    #                 for i in range(len(emotions)):
    #                     r = {"label": emotions[i], "probability": round(preds[j][0][i]*100, 2)}
    #                     data["predictions"].append(r)
    #         data["success"] = True
    #         import json
    #         data_name = f.filename.split('.')[0]+".json"
    #         #os.path.join(basepath,'result', data_name)
    #         with open(data_name, 'w') as outfile:
    #             json.dump(data, outfile)
    #     return jsonify(data)
    # # return render_template('upload.html')


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	#load_model()
	app.run()
