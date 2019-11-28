# *******************************************************************
#
# Author : Thanh Nguyen, 2018
# Email  : sthanhng@gmail.com
# Github : https://github.com/sthanhng
#
# BAP, AI Team
# Face detection using the YOLOv3 algorithm
#
# Description : yoloface.py
# The main code of the Face detection using the YOLOv3 algorithm
#
# *******************************************************************

# Usage example:  python yoloface.py --image samples/outside_000001.jpg \
#                                    --output-dir outputs/
#                 python yoloface.py --video samples/subway.mp4 \
#                                    --output-dir outputs/
#                 python yoloface.py --src 1 --output-dir outputs/

import time
from pygame import mixer # Load the required library
import argparse
import sys
import os
import lir500sa64 as lir500sa
from utils import *

#####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')

# parser.add_argument('--model-weights', type=str,
#                     default='./model-weights/yolov3-face_best.weights',
#                     help='path to weights of model')

parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-wider_16000.weights',
                    help='path to weights of model')

parser.add_argument('--output-dir', type=str, default='outputs/',
                    help='path to the output directory')

args = parser.parse_args()

#####################################################################
# print the arguments
print('----- info -----')
print('[i] The config file: ', args.model_cfg)
print('[i] The weights of model file: ', args.model_weights)
print('###########################################################\n')



# 음악재생용 mixer 초기화.
mixer.init()
mixer.music.load('sound_effect.wav')
mixer.music.set_volume(1)


# check outputs directory
if not os.path.exists(args.output_dir):
    print('==> Creating the {} directory...'.format(args.output_dir))
    os.makedirs(args.output_dir)
else:
    print('==> Skipping create the {} directory...'.format(args.output_dir))

# Give the configuration and weight files for the model and load the network
# using them.
net = cv2.dnn.readNetFromDarknet(args.model_cfg, args.model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)


def _main():
    wind_name = 'face detection using YOLOv3'
    cv2.namedWindow(wind_name, cv2.WINDOW_NORMAL)

    output_file = ''

    # Get data from the camera
    cam = lir500sa.LIR500SA(640, 480)
    ret = cam.connect(b'192.168.254.68')

    if ret:
        print('열화상 카메라와 연결되었습니다.')
        # cam.set_range(25, 80)  # 35온도 설정

    
    # email 객체 생성.
    send_email = sendEmail()

    email_flag = 0


    while True:

        # ret, frame = open(args.video)
        frame = cam.get_frame()
        

        if frame is None:
            print("Disconnected camera")
        elif (frame.size > 0):
           
            cv2.imshow(wind_name, frame)

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                print('[i] ==> Interrupted by user!')
                break
                
    cv2.destroyAllWindows()

    print('==> All done!')
    print('***********************************************************')


if __name__ == '__main__':
    _main()
