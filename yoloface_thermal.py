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
from concurrent.futures import ThreadPoolExecutor
#####################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--model-cfg', type=str, default='./cfg/yolov3-face.cfg',
                    help='path to config file')

parser.add_argument('--model-weights', type=str,
                    default='./model-weights/yolov3-face_final.weights',
                    help='path to weights of model')

# parser.add_argument('--model-weights', type=str,
#                     default='./model-weights/yolov3-wider_16000.weights',
#                     help='path to weights of model')

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
pool_executor = ThreadPoolExecutor(1)
send_email = sendEmail()

def email_processing():

    mixer.music.play()
    time.sleep(1)
    send_email.setContents()
    send_email.sendImage()



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
        cam.set_range(-25, 40)  # 온도 설정

    
    # email 객체 생성.
    # send_email = sendEmail()


    while True:

        # ret, frame = open(args.video)
        frame = cam.get_frame()
        frame = cv2.flip(frame, 1)

        if frame is None:
            print("Disconnected camera")
        elif (frame.size > 0):
            cv2.putText(frame, "Press [ESC] to EXIT...", (5,16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1, cv2.LINE_AA)
            cv2.putText(frame, "Press [ESC] to EXIT...", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA)

            # Create a 4D blob from a frame.

            # frame = np.array([frame, frame, frame])
            # frame = frame.reshape(480,640,3)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
            blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT), [0,0,0], 1, crop=False)
            # blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)
            # blob = cv2.dnn.blobFromImage(frame, 1/255, (480, 640),[0, 0, 0], False, crop=False)
            

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(get_outputs_names(net))

            # Remove the bounding boxes with low confidence
            faces, email_flag = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
            print('[i] ==> # detected faces: {}'.format(len(faces))) # len을 하면 [[790, 200, 109, 119], [287, 174, 106, 113]] 이런식의 숫자를 세준다.
            print('#' * 60)



            # 실험
            #print("final 박스가 도대체 뭘까요?", faces) # face에 box 좌표들이 저장되어있다 즉, 얼굴이 2개면 행렬로 구분되어있다. 
            print("frame은 도대체 어떻게 값이 나올까요?", frame.shape)  # (828, 1168, 3)의 3채널로 이미지를 받아온다
            print("frame 최고값", np.max(frame))  # Numpy 배열 연산만 가능하며, 최고값은 255이다.
            print("frame 최하값", np.min(frame))  # 최하값은 0이다.
            

            # initialize the set of information we'll displaying on the frame
            info = [
                ('number of faces detected', '{}'.format(len(faces)))
            ]

            for (i, (txt, val)) in enumerate(info):
                text = '{}: {}'.format(txt, val)
                cv2.putText(frame, text, (10, (i * 20) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

            # 이메일 보냄.
            
            if (email_flag):
                email_flag = 0
                pool = pool_executor.submit(email_processing)
                # mixer.music.play()
                # time.sleep(1)
                # send_email.setContents()
                # send_email.sendImage()
            
            # cv2.flip(frame, 0)
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







    
