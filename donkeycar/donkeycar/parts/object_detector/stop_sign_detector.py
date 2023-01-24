import numpy as np
import cv2
import time
import random
import collections
from PIL import Image
from matplotlib import cm
import os
import urllib.request
import tensorflow as tf


class StopSignDetector(object):
    '''
    Requires an EdgeTPU for this part to work

    This part will run a EdgeTPU optimized model to run object detection to detect a stop sign.
    We are just using a pre-trained model (MobileNet V2 SSD) provided by Google.
    '''

    model_path = "/home/pi/ADL-Minicar-Challenge-2023/mycar/models/stop_sign_detector_v2.tflite"
    threshold = 0.7

    def __init__(self, max_reverse_count=0, reverse_throttle=-0.5):
        # model related
        #self.model = tf.keras.models.load_model(model_path)
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # reverse throttle related
        self.max_reverse_count = max_reverse_count
        self.reverse_count = max_reverse_count
        self.reverse_throttle = reverse_throttle
        self.is_reversing = False

    def model_predict(self, input_data):
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_details[0]['index'])

    def apply_normalization(self, cv_image_rgb_uint8):
        mean, std = (
            np.array([0.4251, 0.4787, 0.4311]),
            np.array([0.2203, 0.2276, 0.2366])
        )
        image = np.float32(cv_image_rgb_uint8) / 255.0
        image -= mean
        image /= std
        return image

    def detect_stop_sign(self, img):
        img = self.apply_normalization(img)
        #prediction = self.model.predict(np.array([img]))
        prediction = self.model_predict(np.array([img]))
        return prediction[0][0] > self.threshold

    def run(self, img_arr, throttle):
        if img_arr is None:
            return throttle

        stop_signs_detected = self.detect_stop_sign(img_arr)
        if stop_signs_detected:
            print(f"Stop sign detected {stop_signs_detected}")

        if stop_signs_detected or self.is_reversing:

            # Set the throttle to reverse within the max reverse count when detected pedestrians on a zebra crossing
            if self.reverse_count < self.max_reverse_count:
                self.is_reversing = True
                self.reverse_count += 1
                return self.reverse_throttle
            else:
                self.is_reversing = False
                return 0
        else:
            self.is_reversing = False
            self.reverse_count = 0
            return throttle