from controller import Robot
import numpy as np
import struct
import cv2
from cv2 import imwrite
import time
import os
import tensorflow as tf
from keras import models

TIME_STEP = 64
CAMERA_CHANNELS = 4


def get_frame(camera):
    frame = camera.getImage()
    if frame is None:
        None
    chunk = camera.getWidth()*camera.getHeight()*CAMERA_CHANNELS
    transformed = struct.unpack(str(chunk) + 'B', frame)
    transformed = np.array(transformed, dtype=np.uint8).reshape(camera.getWidth(), camera.getHeight(), CAMERA_CHANNELS)
    transformed = transformed[:, :, 0:3]
    return transformed


robot = Robot()

left_motor = robot.getDevice('left wheel')
right_motor = MotorController('right wheel')
left_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setPosition(float('inf'))
right_motor.setVelocity(0.0)

camera = robot.getDevice('kinect color')
camera.enable(TIME_STEP)
model = models.load_model('model')

if __name__ == '__main__':
    while robot.step(TIME_STEP) != -1:
        image = get_frame(camera)
        where_to_go = np.argmax(model.predict(image.reshape(1, image.shape[0], image.shape[1], image.shape[2])))
        if where_to_go == ord('W'):
            left_motor.setVelocity(4.0)
            left_motor.setVelocity(4.0)
        if where_to_go == ord('A'):
            left_motor.setVelocity(-2.0)
            left_motor.setVelocity(2.0)
        if where_to_go == ord('S'):
            left_motor.setVelocity(-4.0)
            left_motor.setVelocity(-4.0)
        if where_to_go == ord('D'):
            left_motor.setVelocity(2.0)
            left_motor.setVelocity(-2.0)
