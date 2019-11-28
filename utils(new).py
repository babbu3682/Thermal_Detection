# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : utils.py
# This file contains the code of the parameters and help functions
#
# *******************************************************************


import datetime
import numpy as np
import cv2

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os


# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------


thermal_threshold = 70 # 온도 36도를 기준으로 짜름. 215

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


        
gmail_sender = "babbu3682@gmail.com"
gmail_receiver = "babbu3682@yonsei.ac.kr"
subject = "Result from Inspection"
email_text = "환자가 발생했습니다, 추가 진단이 필요합니다."
gmail_pwd = 'keuzdpswnfwqokbs'


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------

# Get the names of the output layers
def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected
    # outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box  여기서 마지막 사각형 틀을 그린다.
def draw_predict(frame, conf, left, top, right, bottom):

    # Draw a bounding box.
    # cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)  # 옐로우 박스를 친다.

    # 이제 여기서 좌표값만큼 슬라이싱 -> 그걸 np.array에서 ravel로 하나의 리스트로 쭉 편다 -> max 포인트로 threshold를 잡는다.

    if ( (frame[left:right, top:bottom] > thermal_threshold).mean() >= 0.05 ):
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_RED, 10)  # 2 빨간 박스를 친다.
        text = 'High, score:{:.2f}'.format(conf) # 확신하는 점수가 들어감.
        
        # 빨간 박스를 쳐주고 바로 이미지 저장.            
        cv2.imwrite(os.path.join('outputs/', 'detected_img.jpg'), frame.astype(np.uint8))
        # global email_flag
        email_flag = 1
                             
    else :
        cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 10)  # 옐로우 박스를 친다.
        text = 'Normal, score:{:.2f}'.format(conf) # 확신하는 점수가 들어감.
        
        # 빨간 박스를 쳐주고 바로 이미지 저장.            
        cv2.imwrite(os.path.join('outputs/', 'detected_img.jpg'), frame.astype(np.uint8))
        # global email_flag
        email_flag = 0


    # Display the label at the top of the bounding box 박스에 라벨이 아닌 확신 점수만 출력하기.
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    top = max(top, label_size[1])

    # 이놈이 이제 위의 score를 영상에 출력되게 하는 것.
    cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                COLOR_WHITE, 1)

    return email_flag
                
  

def post_process(frame, outs, conf_threshold, nms_threshold):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only  # 가장 높은 박스 한개만을 출력.
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    email = 0
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
                               nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        final_boxes.append(box)
        left, top, right, bottom = refined_box(left, top, width, height) # 여기서 진정한 딱 1개의 정확한  box 정보를 넘겨준다.
        # draw_predict(frame, confidences[i], left, top, left + width,
        #              top + height)
        email = draw_predict(frame, confidences[i], left, top, right, bottom)
    return final_boxes, email


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._num_frames = 0

    def start(self):
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._num_frames += 1

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._num_frames / self.elapsed()

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom



class sendEmail:
    def __init__(self):
        self.msg = MIMEMultipart('alternative')
        self.msg['Subject'] = subject
        self.msg['From'] = gmail_sender
        self.msg['To'] = gmail_receiver
        
    def setContents(self): 

        self.msg.attach(MIMEText(email_text, _charset='utf-8'))  # 메일 내용 attach 
        
        attach_img = 'outputs/' + 'detected_img.jpg' # jpg의 경로 

        part = MIMEBase('application','octet-stream')
        part.set_payload(open(attach_img,'rb').read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', 'attachment; filename=%s'%os.path.basename(attach_img))
        
        self.msg.attach(part) # 사진을 attach 첨부

    
    def sendImage(self):
        mailServer = smtplib.SMTP('smtp.gmail.com', 587)
        mailServer.ehlo()
        mailServer.starttls() 
        mailServer.login(gmail_sender, gmail_pwd)
        mailServer.sendmail(gmail_sender, gmail_receiver, self.msg.as_string()) # 오류
        mailServer.close()



