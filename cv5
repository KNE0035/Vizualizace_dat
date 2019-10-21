import numpy as np
import math
import cv2
from os import listdir
import random

def main():
    dirname = "./flow_256/"
    files = listdir(dirname)

    for file in files:
        fs = cv2.FileStorage(dirname + file, cv2.FILE_STORAGE_READ)
        fn = fs.getNode("flow")

        vectorNpData = np.asarray(fn.mat())
        xVectorNpData, yVectorNpData = cv2.split(vectorNpData)
        grad_x = cv2.Sobel(xVectorNpData, cv2.CV_32FC1, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(yVectorNpData, cv2.CV_32FC1, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

        grad_y_by_x = cv2.Sobel(yVectorNpData, cv2.CV_32FC1, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_x_by_y = cv2.Sobel(xVectorNpData, cv2.CV_32FC1, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

        rotMagnitudeImg = grad_y_by_x - grad_x_by_y

        divergenceImg = grad_x + grad_y

        height, width = divergenceImg.shape
        divergenceheatMap = np.zeros((width, height, 3), np.uint8)
        rotMagnitudeHeatMap = np.zeros((width, height, 3), np.uint8)

        for y in range(0, height):
            for x in range(0, width):
                ramp = [[210, 210, 255], [0, 0, 255], [0, 0, 0], [255, 0, 0], [255, 210, 210]]
                divergenceheatMap[x][y] = colorRamp(divergenceImg[x][y], divergenceImg.min(), divergenceImg.max(), ramp, 5)
                rotMagnitudeHeatMap[x][y] = colorRamp(rotMagnitudeImg[x][y], rotMagnitudeImg.min(), rotMagnitudeImg.max(), ramp, 5)

        cv2.imshow("divergence", divergenceheatMap)
        cv2.imshow("rotation magnitude", rotMagnitudeHeatMap)
        cv2.waitKey(1)
        print(dirname + file)


def colorRamp(value, min_value, max_value, ramp, ramp_size):
    if value >= max_value:
        return ramp[ramp_size - 1]
    else:
        a = (value - min_value) / ((max_value - min_value) / (ramp_size - 1))
        band = int((math.floor(a)))
        a -= band
        r = ramp[band][0] * (1 - a) + ramp[band + 1][0] * a
        g = ramp[band][1] * (1 - a) + ramp[band + 1][1] * a
        b = ramp[band][2] * (1 - a) + ramp[band + 1][2] * a

        return [r, g, b]


if __name__ == "__main__":
    main()