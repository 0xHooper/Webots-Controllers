from controller import Robot
from controller import Keyboard
import numpy as np
import struct
import cv2
from cv2 import imwrite
import time
import os


class PDcontroller:
    def __init__(self, p, d, sampling_period, target=0.0):
        self.target = target
        self.response = 0.0
        self.old_error = 0.0
        self.p = p
        self.d = d
        self.sampling_period = sampling_period

    def process_measurement(self, measurement):
        error = self.target - measurement
        derivative = (error - self.old_error)/self.sampling_period
        self.old_error = error
        self.response = self.p*error + self.d*derivative
        return self.response

    def reset(self):
        self.target = 0.0
        self.response = 0.0
        self.old_error = 0.0


class MotorController:

    def __init__(self, name, pd):
        self.name = name
        self.pd = pd
        self.motor = None
        self.velocity = 0.0

    def enable(self):
        self.motor = robot.getDevice(self.name)
        self.motor.setPosition(float('inf'))
        self.motor.setVelocity(0.0)

    def update(self):
        self.velocity += self.pd.process_measurement(self.motor.getVelocity())
        self.motor.setVelocity(self.velocity)

    def set_target(self, target):
        self.pd.target = target

    def emergency_break(self):
        self.motor.setVelocity(0.0)
        self.pd.reset()


class MotorCommand:

    def __init__(self, left_velocity, right_velocity, emergency_break=False):
        self.left_velocity = left_velocity
        self.right_velocity = right_velocity
        self.emergency_break = emergency_break


# CONSTANTS
CRUISING_SPEED = 5.0
TURN_SPEED = CRUISING_SPEED/2.0
TIME_STEP = 64
TIME_STEP_SECONDS = TIME_STEP/1000
CAMERA_CHANNELS = 4

motor_commands = {
        ord('W'): MotorCommand(CRUISING_SPEED, CRUISING_SPEED),
        ord('X'): MotorCommand(-CRUISING_SPEED, -CRUISING_SPEED),
        ord('A'): MotorCommand(-TURN_SPEED, TURN_SPEED),
        ord('D'): MotorCommand(TURN_SPEED, -TURN_SPEED),
        ord('S'): MotorCommand(0.0, 0.0),
        ord('E'): MotorCommand(0.0, 0.0, True)
}


def get_frame(camera):
    frame = camera.getImage()
    if frame is None:
        None
    chunk = camera.getWidth()*camera.getHeight()*CAMERA_CHANNELS
    transformed = struct.unpack(str(chunk) + 'B', frame)
    transformed = np.array(transformed, dtype=np.uint8).reshape(camera.getWidth(), camera.getHeight(), CAMERA_CHANNELS)
    transformed = transformed[:, :, 0:3]
    return transformed


# INITIALIZATION
robot = Robot()
keyboard = Keyboard()
keyboard.enable(TIME_STEP)

left_motor = MotorController('left wheel', PDcontroller(0.01, 0.0001, TIME_STEP_SECONDS))
right_motor = MotorController('right wheel', PDcontroller(0.01, 0.0001, TIME_STEP_SECONDS))
left_motor.enable()
right_motor.enable()

camera = robot.getDevice('kinect color')
camera.enable(TIME_STEP)


def process_keyboard_control(key):
    if key in motor_commands.keys():
        if motor_commands[key].emergency_break:
            left_motor.emergency_break()
            right_motor.emergency_break()
        else:
            left_motor.set_target(motor_commands[key].left_velocity)
            right_motor.set_target(motor_commands[key].right_velocity)


def get_files_number(dir_path):
    for base, dirs, files in os.walk(dir_path):
        return len(files)


def save_image(save_path):
    index = get_files_number(save_path) + 1
    frame = get_frame(camera)
    file_name = save_path + str(index) + '.png'
    cv2.imwrite(file_name, frame)


if __name__ == '__main__':
    while robot.step(TIME_STEP) != -1:
        left_motor.update()
        right_motor.update()
        pressed_key = keyboard.getKey()
        process_keyboard_control(pressed_key)
        if pressed_key == Keyboard.SHIFT+ord('W'):
            path = 'photo/w/'
            save_image(path)
        if pressed_key == Keyboard.SHIFT+ord('A'):
            path = 'photo/a/'
            save_image(path)
        if pressed_key == Keyboard.SHIFT+ord('S'):
            path = 'photo/s/'
            save_image(path)
        if pressed_key == Keyboard.SHIFT+ord('D'):
            path = 'photo/d/'
            save_image(path)

