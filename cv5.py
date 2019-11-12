from decimal import Decimal

import numpy as np
import math
import cv2
from os import listdir
import random

def main():
    dirname = "./flow_256/"
    files = listdir(dirname)
    randomPoint = (random.randint(0, 255), random.randint(0, 255))
    randomPoint = (65, 51)
    ramp = [[210, 210, 255], [0, 0, 255], [0, 0, 0], [255, 0, 0], [255, 210, 210]]
    noRandomPoints = 20
    randomPoints = []
    boundaries = cv2.imread("boundaries_256.png", 0)

    for y in range(0, 255):
        for x in range(0, 255):
            if boundaries[x][y] != 0:
                print(x, y)

    #cv2.imshow("asf", boundaries)
    #cv2.waitKey(0)

    for i in range(noRandomPoints):
        randomPoint = (random.randint(0, 255), random.randint(0, 255))
        randomPoints.append(randomPoint)

    #randomPoints.append((128, 180))

    for file in files:
        fs = cv2.FileStorage(dirname + file, cv2.FILE_STORAGE_READ)
        fn = fs.getNode("flow")

        vectorNpData = np.asarray(fn.mat())
        height, width = vectorNpData.shape[:2]
        xVectorNpData, yVectorNpData = cv2.split(vectorNpData)
        #grad_x = cv2.Sobel(xVectorNpData, cv2.CV_32FC1, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        #grad_y = cv2.Sobel(yVectorNpData, cv2.CV_32FC1, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

        #grad_y_by_x = cv2.Sobel(yVectorNpData, cv2.CV_32FC1, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        #grad_x_by_y = cv2.Sobel(xVectorNpData, cv2.CV_32FC1, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

        #rotMagnitudeImg = grad_y_by_x - grad_x_by_y
        #divergenceImg = grad_x + grad_y
        speedMapImg = np.zeros((width, height, 1), np.float)
        speedHeatMap = np.zeros((width, height, 3), np.uint8)
        for y in range(0, height):
            for x in range(0, width):
                speedMapImg[x][y] = math.sqrt(float(xVectorNpData[x][y]) * float(xVectorNpData[x][y]) + float(yVectorNpData[x][y]) * float(yVectorNpData[x][y]))

        divergenceheatMap = np.zeros((width, height, 3), np.uint8)
        rotMagnitudeHeatMap = np.zeros((width, height, 3), np.uint8)

        #for y in range(0, height):
            #for x in range(0, width):
                #speedHeatMap[x][y] = colorRamp(speedMapImg[x][y], speedMapImg.min(), speedMapImg.max(), ramp, 5)
                #divergenceheatMap[x][y] = colorRamp(divergenceImg[x][y], divergenceImg.min(), divergenceImg.max(), ramp, 5)
                #rotMagnitudeHeatMap[x][y] = colorRamp(rotMagnitudeImg[x][y], rotMagnitudeImg.min(), rotMagnitudeImg.max(), ramp, 5)


        for i in range(noRandomPoints):
            vectorOnRandomPoint = vectorNpData[randomPoints[i][0]][randomPoints[i][1]];
            shiftedPoint = (int(randomPoints[i][0] + vectorOnRandomPoint[0] * 10), int(randomPoints[i][1] + vectorOnRandomPoint[1] * 10))

            if shiftedPoint[0] in range(0, width) and shiftedPoint[1] in range(0, height) and boundaries[shiftedPoint[1]][shiftedPoint[0]] != 255:
                randomPoints[i] = shiftedPoint

            speedMapImg = cv2.circle(speedMapImg, randomPoints[i], 3, 255, -1)


        cv2.imshow("speed map img", speedMapImg)
        #cv2.imshow("rotation magnitude", rotMagnitudeHeatMap)
        cv2.waitKey(1)

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