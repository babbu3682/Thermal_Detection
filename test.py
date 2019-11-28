import cv2
import numpy as np

cap = cv2.imread('samples/sad (2).jpg')

np.savetxt("0번층", cap[:,:,0], fmt="%d")

np.savetxt("1번층", cap[:,:,1], fmt="%d")


np.savetxt("2번층", cap[:,:,2], fmt="%d")

print("0번층\n", cap[:,:,0])
print("1번층\n", cap[:,:,1])
print("2번층\n", cap[:,:,2])
print("형태", cap.shape)

print("0과1",np.array_equal(cap[:,:,0],cap[:,:,1]))
print("0과2",np.array_equal(cap[:,:,0],cap[:,:,2]))
print("1과2",np.array_equal(cap[:,:,1],cap[:,:,2]))