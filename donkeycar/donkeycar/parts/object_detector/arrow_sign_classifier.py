import numpy as np
import cv2
import time
import random
from collections import deque
from PIL import Image
from matplotlib import cm
import os
import urllib.request
import tensorflow as tf


class ArrowSignClassifier(object):
    '''
    Class for arrow sign classifier
    '''

    model_path = "/home/pi/ADL-Minicar-Challenge-2023/mycar/models/arrow_sign_classifier.tflite"
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

    def detect_arrow_sign(self, img):
        img = self.apply_normalization(img)
        img = img[160:-160, ...]
        prediction = self.model_predict(np.array([img]))
        return prediction[0][0] > self.threshold

    def run(self, img_arr, angle, previous_arrow_detections=None):
        if img_arr is None:
            return angle, previous_arrow_detections

        if angle is None:
            angle = 0

        turn_right = self.detect_arrow_sign(img_arr)

        if previous_arrow_detections is None:
            previous_arrow_detections = deque([turn_right])

        if 1 < len(previous_arrow_detections) < 5:
            previous_arrow_detections.append(turn_right)
        else:
            previous_arrow_detections.popleft()
            previous_arrow_detections.append(turn_right)

        if sum(previous_arrow_detections) == 5:
            turn_right = True
        elif sum(previous_arrow_detections) == 0:
            turn_right = False

        if turn_right:
            print(f"Turn to the right!")
            angle += 0.2
        else:
            print(f"Turn to the left!")
            angle -= 0.2

        return angle, previous_arrow_detections