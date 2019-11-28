import lir500sa64 as lir500sa
import cv2

cam = lir500sa.LIR500SA(640, 480)
ret = cam.connect(b'192.168.254.68')
if ret:
    print('Connected')
    cam.set_range(25, 38)
    
    while True:
        img = cam.get_frame()
        if img is None:
            print("Disconnected camera")
        elif img.size > 0:
            cv2.putText(img, "Press [ESC] to EXIT...", (5,16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1, cv2.LINE_AA)
            cv2.putText(img, "Press [ESC] to EXIT...", (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1, cv2.LINE_AA)

            print("이미지 shape=", img.shape)
            cv2.imshow('camera', img)
        
        if cv2.waitKey(1) == 27:
            break
    
    cam.disconnect()

else:
    print("Can't connect camera")
