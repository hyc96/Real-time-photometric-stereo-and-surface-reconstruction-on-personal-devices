import cv2
import time
import sys


if __name__ == "__main__":
    a = 0
    if len(sys.argv) != 2:
        print("usage: python main.py [option number]\n 1: capture 4 images, 2: generate 1 frame of height map, 3: real-time stereo, 4: display real-time results\n")
        exit()
    else:
        [placeholder, a] = sys.argv
        a = int(a)

    if a > 0 and a < 5:
        from util import *

    if a == 1:
        get_image()
    elif a == 2:
        I, size = get_input()
        surface_normal(I, size)
    elif a == 3:
        real_time_stereo()
    elif a == 4:
        display_frames()
    else:
        print("Please input a valid number")