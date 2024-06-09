import cv2
from matplotlib import pyplot as plt
import time

i = 0
retu = False
while (retu == False):
    cap = cv2.VideoCapture(i)

    ret, frame = cap.read()

    retu = ret

    if (retu == True):

        print("Value of i is: ", i)

    cap.release()

    i += 1
