from collections import defaultdict

import numpy as np
import math
import cv2
import random

def signalFunction(point, scale):
    return math.sin(point[0] * scale) * math.cos(point[1] * scale)

def listen(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        drawing = False

        print("x: {}, y: {}".format(x, y))

def main():

    stepx = 10
    stepy = 10
    scale = 0.01
    size = (int(2 * (math.pi) / scale) + 1, int(2 * (math.pi) / scale) + 1, 1)

    residue = size[0] % stepx

    size = (size[0] - residue + 1, size[1] - residue + 1)


    signalImg = np.zeros(size, dtype="uint8")

    cv2.namedWindow('image', cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('image', listen, [signalImg])

    signalImgReference = np.zeros(size, dtype="uint8")
    regularGrid = False
    randomRangex = int(stepx / 3)
    randomRangey = int(stepy / 3)
    randomShiftx = 0
    randomShifty = 0
    samplePointsWithQuadrants = defaultdict(list)

    noGridx = int(size[0] / stepx)
    noGridy = int(size[1] / stepy)

    for yGridPart in range(0, noGridy):
        for xGridPart in range(0, noGridx):
            keyGridPart = (xGridPart, yGridPart)

            leftUpperPoint = (stepx * xGridPart, stepy * yGridPart)
            rightUpperPoint = (stepx * xGridPart + stepx, stepy * yGridPart)
            leftDownPoint = (stepx * xGridPart, stepy * yGridPart + stepy)
            rightDownPoint = (stepx * xGridPart + stepx, stepy * yGridPart + stepy)

            if not regularGrid:
                leftUpperPoint = calculateRandomShiftWithRespectToBoundaries(False, False, leftUpperPoint, size, randomRangex, randomRangey)
                rightUpperPoint = calculateRandomShiftWithRespectToBoundaries(True, False, rightUpperPoint, size, randomRangex, randomRangey)
                leftDownPoint = calculateRandomShiftWithRespectToBoundaries(False, True, leftDownPoint, size, randomRangex, randomRangey)
                rightDownPoint = calculateRandomShiftWithRespectToBoundaries(True, True, rightDownPoint, size, randomRangex, randomRangey)

            signalImg[leftUpperPoint] = (signalFunction(leftUpperPoint, scale) + 1) * 127
            signalImg[rightUpperPoint] = (signalFunction(rightUpperPoint, scale) + 1) * 127
            signalImg[leftDownPoint] = (signalFunction(leftDownPoint, scale) + 1) * 127
            signalImg[rightDownPoint] = (signalFunction(rightDownPoint, scale) + 1) * 127

            samplePointsWithQuadrants[keyGridPart].append(leftUpperPoint);
            samplePointsWithQuadrants[keyGridPart].append(rightUpperPoint);
            samplePointsWithQuadrants[keyGridPart].append(leftDownPoint);
            samplePointsWithQuadrants[keyGridPart].append(rightDownPoint);

    cv2.imshow("sampled", signalImg)
    cv2.waitKey()

    for y in range(0, size[1]):
        for x in range(0, size[0]):
            signalImgReference[x, y] = (signalFunction((x, y), scale) + 1) * 127

    for y in range(0, size[1] - 1):
        for x in range(0, size[0] - 1):
            yGridPart = int(y / stepy);
            xGridPart = int(x / stepx);
            keyGridPart = (xGridPart, yGridPart)

            if (x, y) in samplePointsWithQuadrants[keyGridPart]:
                continue

            leftUpperPoint = samplePointsWithQuadrants[keyGridPart][0]
            rightUpperPoint = samplePointsWithQuadrants[keyGridPart][1]
            leftDownPoint = samplePointsWithQuadrants[keyGridPart][2]
            rightDownPoint = samplePointsWithQuadrants[keyGridPart][3]

            if(regularGrid):
                bilinearInterpolationForXY(signalImg, leftUpperPoint, leftDownPoint, rightUpperPoint, rightDownPoint, x, y)
            else:
                interpolateIregularGrid(signalImg, leftUpperPoint, leftDownPoint, rightUpperPoint, rightDownPoint, x, y)

    cv2.imshow("image", signalImg)
    cv2.imshow("reference", signalImgReference)
    cv2.waitKey()

def createJacobian(r, s, p1, p2, p3, p4):
    # dt_dr = s * (p3 -p4) + (1 - s) * (-p1 + p2)
    # dt_ds - -p1*(1-r) - p2*r + p3*r + p4*(1-r)

    jacobian = np.zeros((2, 2), dtype="float64")
    jacobian[0, 0] = s * (p3[0] - p4[0]) + (1 - s) * (-p1[0] + p2[0])
    jacobian[0, 1] = s * (p3[1] - p4[1]) + (1 - s) * (-p1[1] + p2[1])

    jacobian[1, 0] = -p1[0] * (1 - r) - p2[0] * r + p3[0] * r + p4[0] * (1 - r)
    jacobian[1, 1] = -p1[1] * (1 - r) - p2[1] * r + p3[1] * r + p4[1] * (1 - r)

    return jacobian

def interpolateIregularGrid(signalImg, leftUpperPoint, leftDownPoint, rightUpperPoint, rightDownPoint, x, y):
    rs = getRSByNewtonsMethod(leftUpperPoint, leftDownPoint, rightUpperPoint, rightDownPoint, x, y, 3)

    r = rs[0, 0]
    s = rs[0, 1]

    f_00 = int(signalImg[leftUpperPoint])
    f_01 = int(signalImg[leftDownPoint])
    f_10 = int(signalImg[rightUpperPoint])
    f_11 = int(signalImg[rightDownPoint])

    signalImg[x, y] = f_01 * (1 - r) * (1 - s) + f_11 * r * (1 - s) + f_10 * r * s + f_00 * (1 - r) * s

def getRSByNewtonsMethod( leftUpperPoint, leftDownPoint, rightUpperPoint, rightDownPoint, x, y, noIteration):
    baseRS = np.zeros((1, 2), dtype="float64")
    baseRS[0,0] = 0.5
    baseRS[0,1] = 0.5

    basePoint = np.zeros((1, 2), np.float64)
    basePoint[0, 0] = x
    basePoint[0, 1] = y

    for x in range(0, noIteration):
        titeratedP= np.zeros((1, 2), np.float64)
        titeratedP[0, 0] = (1 - baseRS[0, 1]) * ((1 - baseRS[0, 0]) * leftDownPoint[0] + baseRS[0, 0] * rightDownPoint[0]) + baseRS[0, 1] * (baseRS[0, 0] * rightUpperPoint[0] + (1 - baseRS[0,0]) * leftUpperPoint[0])
        titeratedP[0, 1] = (1 - baseRS[0, 1]) * ((1 - baseRS[0, 0]) * leftDownPoint[1] + baseRS[0, 0] * rightDownPoint[1]) + baseRS[0, 1] * (baseRS[0, 0] * rightUpperPoint[1] + (1 - baseRS[0, 0]) * leftUpperPoint[1])
        jacobian = createJacobian(baseRS[0, 0], baseRS[0, 1], leftDownPoint, rightDownPoint, rightUpperPoint, leftUpperPoint)
        invertedJacobian = np.zeros((2, 2), dtype="float64")
        cv2.invert(jacobian, invertedJacobian, cv2.DECOMP_LU)

        baseRS = baseRS - invertedJacobian * (titeratedP - basePoint)

    return baseRS



def bilinearInterpolationForXY(signalImg, leftUpperPoint, leftDownPoint, rightUpperPoint, rightDownPoint, x, y):
    f_00 = float(signalImg[leftUpperPoint])
    f_01 = float(signalImg[leftDownPoint])
    f_10 = float(signalImg[rightUpperPoint])
    f_11 = float(signalImg[rightDownPoint])
    x_ = x - leftUpperPoint[0]
    y_ = y - leftUpperPoint[1]

    Rx1 = ((rightUpperPoint[0] - x) / (rightUpperPoint[0] - leftUpperPoint[0])) * f_00 + ((x - leftUpperPoint[0]) / (rightUpperPoint[0] - leftUpperPoint[0])) * f_10;
    Rx2 = ((rightUpperPoint[0] - x) / (rightUpperPoint[0] - leftUpperPoint[0])) * f_01 + ((x - leftUpperPoint[0]) / (rightUpperPoint[0] - leftUpperPoint[0])) * f_11;

    value = ((rightUpperPoint[1] - y) / (rightUpperPoint[1] - rightDownPoint[1])) * Rx2 + ((y - leftDownPoint[1]) / (rightUpperPoint[1] - rightDownPoint[1])) * Rx1;

    signalImg[x, y] = value

def calculateRandomShiftWithRespectToBoundaries(shiftPositivex, shiftPositivey, point, boundaries, randomRangex, randomRangey):
    randomShiftx = random.randint(0, randomRangex) if shiftPositivex else random.randint(-randomRangex, 0)
    randomShifty = random.randint(0, randomRangey) if shiftPositivey else random.randint(-randomRangey, 0)

    newPoint = (point[0] + randomShiftx, point[1] + randomShifty)
    if newPoint[0] < 0 or newPoint[0] >= boundaries[0] or newPoint[1] < 0 or newPoint[1] >= boundaries[1]:
        return point
    return newPoint


if __name__ == "__main__":
    main()