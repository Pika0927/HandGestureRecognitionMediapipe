#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import csv
import copy

import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import pyautogui as pag

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from model import HandHistoryClassifier
from model import LeftKeyPointClassifier
from model import RightKeyPointClassifier

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1080)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    args = parser.parse_args()

    return args


def main():
    # Argument parsing ################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ##############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load ######################################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    leftkeypoint_classifier = LeftKeyPointClassifier()
    rightkeypoint_classifier = RightKeyPointClassifier()

    # Read labels ###################################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # FPS Measurement ###########################################################

    #  ########################################################################
    handCounter = 0
    InputText = ""
    OKCounter = 0
    ClearCounter = 0
    PreHandSign = -1
    OKcounter = 0
    while True:
        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                handedness_index = 0
                if handedness.classification[0].label == 'Left':
                    handedness_index = 0
                    handCounter = 1
                elif handedness.classification[0].label == 'Right':
                    handedness_index = 1
                    handCounter += 1
                # Conversion to relative coordinates / normalized coordinates

                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)

                # Hand sign classification
                if (handedness_index == 0):
                    continue
                    hand_sign_id, sign_fit = leftkeypoint_classifier(
                        pre_processed_landmark_list)
                    if (hand_sign_id == 0):
                        OKCounter += 1
                        ClearCounter = 0
                    elif (hand_sign_id == 1):
                        OKCounter = 0
                        ClearCounter += 1
                    else:
                        OKCounter = 0
                        ClearCounter = 0
                else:
                    hand_sign_id, sign_fit = rightkeypoint_classifier(
                        pre_processed_landmark_list)
                    if(sign_fit < 0.5):
                        PreHandSign = -1
                        OKcounter = 0
                    elif(PreHandSign != hand_sign_id):
                        PreHandSign = hand_sign_id
                    elif(PreHandSign == hand_sign_id):
                        OKcounter += 1

                    if (OKcounter == 7 and hand_sign_id < 10):
                        pag.press("num"+str(hand_sign_id))
                        InputText += str(hand_sign_id)
                        OKcounter = 0
                    elif (ClearCounter == 15):
                        InputText = ""

                # Drawing part

                # debug_image = draw_input_text(
                #     debug_image,
                #     InputText,
                # )

        # Screen reflection #############################################################
        # cv.imshow('Hand Gesture Recognition', debug_image)
    cap.release()
    cv.destroyAllWindows()


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def draw_input_text(image, inputtext):
    cv.putText(image, inputtext, (10, 120), cv.FONT_HERSHEY_SIMPLEX, 2,
               (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, inputtext, (10, 120), cv.FONT_HERSHEY_SIMPLEX, 2,
               (255, 255, 50), 2, cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
