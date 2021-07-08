"""redBallFollower controller."""

from controller import Robot
import numpy as np
import struct
import cv2
import time


# CONSTANTS
TIME_STEP = 64
CAMERA_CHANNELS = 4
MAX_VELOCITY = 7


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
left_motor = robot.getDevice('lMotor')
right_motor = robot.getDevice('rMotor')
camera = robot.getDevice('cam1')
camera.enable(TIME_STEP)
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
sonar = robot.getDevice('sonar')
sonar.enable(TIME_STEP)
last_seen = 0


def move():
    middle = last_seen
    if len(conts) > 0:
        x, y, w, h = cv2.boundingRect(conts[0])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 102), 2)
        middle = x + w / 2
        if middle < 28:
            left_motor.setVelocity(-2.0)
            right_motor.setVelocity(2.0)
        elif middle > 32:
            left_motor.setVelocity(2.0)
            right_motor.setVelocity(-2.0)
        elif sonar.getValue() < 1000:
            left_motor.setVelocity(0.0)
            right_motor.setVelocity(0.0)
        else:
            left_motor.setVelocity(4.0)
            right_motor.setVelocity(4.0)
    else:
        if last_seen < 28:
            where_to_turn = -1.0
        else:
            where_to_turn = 1.0
        left_motor.setVelocity(where_to_turn)
        right_motor.setVelocity(-where_to_turn)
    return middle


def create_red_mask():
    mask_lower = cv2.inRange(frame_hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    mask_upper = cv2.inRange(frame_hsv, np.array([170, 80, 80]), np.array([180, 255, 255]))
    return cv2.bitwise_or(mask_lower, mask_upper)


# MAIN LOOP
while robot.step(TIME_STEP) != -1:
    right_motor.setVelocity(0.0)
    left_motor.setVelocity(0.0)
    frame = get_frame(camera)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)

    # creating mask needed for red color detection
    mask = create_red_mask()
    # I haven't added filtering due to not detecting red color from larger distances
    conts, h = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # robot's camera sometimes doesn't detect red ball, I had to add last_seen value,
    # to help him get to the target faster (robot is still quite confused)
    last_seen = move()
    cv2.imshow("mask", mask)
    cv2.imshow('cam', frame)
    cv2.waitKey(1)
