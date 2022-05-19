#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%
import csv
import copy

import argparse
import itertools
from collections import Counter
from collections import deque
from math import dist

import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import pyautogui as pag
import struct

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from model import HandHistoryClassifier
from model import LeftKeyPointClassifier
from model import RightKeyPointClassifier
from datetime import datetime

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

    print("Start GNI")

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    cvFpsCalc = CvFpsCalc(buffer_len=10)
    # Camera preparation ##############################################################
    start = time.time()
    cap = cv.VideoCapture(cap_device)
    print('Camera: %.2f sec' % (time.time() - start))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    prex = -1
    prey = -1
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

    # FPS Measurement #########################################################

    #  ########################################################################
    handCounter = 0
    InputText = ""
    OKCounter = 0
    ClearCounter = 0
    SameCounter = 0
    PreRightHandTag = -1
    ScreenX,ScreenY = pag.size()
    MouseDown = 0

    f = open(r'\\.\pipe\NPGesture', 'r+b', 0)

    while True:
        fps = cvFpsCalc.get()
        print("fps = "+str(fps))
        # Process Key (ESC: end) ##############################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # Camera capture ######################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        # Detection implementation ############################################
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
                #draw_landmarks(debug_image,landmark_list)
                # Hand sign classification
                if (handedness_index == 0):
                    hand_sign_id, sign_fit = leftkeypoint_classifier(
                        pre_processed_landmark_list)
                else:
                    hand_sign_id, sign_fit = rightkeypoint_classifier(
                        pre_processed_landmark_list)
                    if(sign_fit>0.5):
                        s=(str(hand_sign_id)+","+
                        str(landmark_list[8][0])+","+
                        str(landmark_list[8][1])+","+
                        str(landmark_list[5][0])+","+
                        str(landmark_list[5][1])
                        ).encode('ascii')
                    else:
                        s='x'.encode('ascii')

                    f.write(struct.pack('I', len(s)) + s)   # Write str length and str
                    #print(datetime.now().time())
                    #print("ID : "+str(hand_sign_id)+" FIT : "+str(sign_fit)+" FPS : "+str(fps)+" ")
                    #f.seek(0)                               # EDIT: This is also necessary
                    '''
                    if (PreRightHandTag == hand_sign_id):
                        SameCounter += 1
                    else:
                        SameCounter = 0
                        PreRightHandTag = hand_sign_id
                    
                    if (SameCounter > 3 and sign_fit > 0.5):
                        
                        if (hand_sign_id == 1):
                            SameCounter += 1
                            posx = landmark_list[8][0]
                            posy = landmark_list[8][1]
                            Rate = ScreenY / 4 / dist(landmark_list[8],landmark_list[5])
                            if (prex == -1):
                                prex = posx
                                prey = posy
                            else:
                                pag.moveRel((posx - prex)*Rate,
                                            (posy - prey)*Rate,
                                            _pause=False)
                                prex = posx
                                prey = posy
                        elif(hand_sign_id == 5 and SameCounter == 5):
                            pag.click()
                            prex = -1
                            prey = -1
                    
                    else:
                        prex = -1
                        prey = -1
                    '''
                    '''
                        if (hand_sign_id == 2):
                            pag.moveRel(-10, 0)
                        if (hand_sign_id == 3):
                            pag.moveRel(0, 10)
                        if (hand_sign_id == 4):
                            pag.moveRel(0, -10)
                        if (hand_sign_id == 11):
                            pag.click()
                    '''
                # Drawing part
                '''
                debug_image = draw_input_text(
                    debug_image,
                    InputText,
                )
                '''

        # Screen reflection #############################################################
        #cv.imshow('Hand Gesture Recognition', debug_image)
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

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
      
    return image

if __name__ == '__main__':
    main()
