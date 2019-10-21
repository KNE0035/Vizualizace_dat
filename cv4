import numpy as np
import math
import cv2
import random
from sklearn.neighbors import NearestNeighbors

def signalFunction(point, scale):
    return math.sin(point[0] * scale) * math.cos(point[1] * scale)

def main():

    noNeigbours = 4
    stepx = 4
    stepy = 4
    scale = 0.1
    k = 4
    size = (int(2 * (math.pi) / scale) + 1, int(2 * (math.pi) / scale) + 1, 1)

    residue = size[0] % stepx
    size = (size[0] - residue + 1, size[1] - residue + 1)

    signalImg = np.zeros(size, dtype="uint8")

    signalImgReference = np.zeros(size, dtype="uint8")
    randomRangex = int(stepx / 3)
    randomRangey = int(stepy / 3)
    randomShiftx = 0
    randomShifty = 0

    noGridx = int(size[0] / stepx)
    noGridy = int(size[1] / stepy)

    sampleArray = np.array([], dtype=np.int32).reshape(0, 2)
    for yGridPart in range(0, noGridy):
        for xGridPart in range(0, noGridx):
            keyGridPart = (xGridPart, yGridPart)

            leftUpperPoint = (stepx * xGridPart, stepy * yGridPart)
            rightUpperPoint = (stepx * xGridPart + stepx, stepy * yGridPart)
            leftDownPoint = (stepx * xGridPart, stepy * yGridPart + stepy)
            rightDownPoint = (stepx * xGridPart + stepx, stepy * yGridPart + stepy)

            leftUpperPoint = calculateRandomShiftWithRespectToBoundaries(False, False, leftUpperPoint, size, randomRangex, randomRangey)
            rightUpperPoint = calculateRandomShiftWithRespectToBoundaries(True, False, rightUpperPoint, size, randomRangex, randomRangey)
            leftDownPoint = calculateRandomShiftWithRespectToBoundaries(False, True, leftDownPoint, size, randomRangex, randomRangey)
            rightDownPoint = calculateRandomShiftWithRespectToBoundaries(True, True, rightDownPoint, size, randomRangex, randomRangey)

            X = np.array([[leftUpperPoint[0], leftUpperPoint[1]],
                         [rightUpperPoint[0], rightUpperPoint[1]],
                         [leftDownPoint[0], leftDownPoint[1]],
                         [rightDownPoint[0], rightDownPoint[1]]])

            sampleArray = np.concatenate((sampleArray, X))

            signalImg[leftUpperPoint] = (signalFunction(leftUpperPoint, scale) + 1) * 127
            signalImg[rightUpperPoint] = (signalFunction(rightUpperPoint, scale) + 1) * 127
            signalImg[leftDownPoint] = (signalFunction(leftDownPoint, scale) + 1) * 127
            signalImg[rightDownPoint] = (signalFunction(rightDownPoint, scale) + 1) * 127

    cv2.imshow("sampled", signalImg)
    cv2.waitKey()
    unq, count = np.unique(sampleArray, axis=0, return_counts=True)
    sampleArray = unq[count > 1]

    knn = NearestNeighbors(n_neighbors=noNeigbours, radius=5.0)
    knn.fit(sampleArray)

    for y in range(0, size[1]):
        for x in range(0, size[0]):
            signalImgReference[x, y] = (signalFunction((x, y), scale) + 1) * 127

    for y in range(0, size[1]):
        for x in range(0, size[0]):
            isSameItem = False
            for item in sampleArray:
                if item[0] == x and item[1] == y:
                    isSameItem = True
                    break

            if isSameItem:
                continue

            distance, neigbours = knn.kneighbors([[x, y]], return_distance=True)

            denominator = 0
            numerator = 0

            length = len(neigbours[0])
            for i in range(length):
                numerator += (1 / distance[0][i]) * signalImg[sampleArray[neigbours[0][i]][0], sampleArray[neigbours[0][i]][1]];
                denominator += (1 / distance[0][i])

            signalImg[x, y] = numerator / denominator




    cv2.imshow("image", signalImg)
    cv2.imshow("reference", signalImgReference)
    cv2.waitKey()

    error = 0
    for y in range(0, size[1]):
        for x in range(0, size[0]):
            error += math.fabs(signalImg[x, y] - signalImgReference[x, y])

    error = error / (size[0] * size[1])
    print(error)

def calculateRandomShiftWithRespectToBoundaries(shiftPositivex, shiftPositivey, point, boundaries, randomRangex, randomRangey):
    randomShiftx = random.randint(0, randomRangex) if shiftPositivex else random.randint(-randomRangex, 0)
    randomShifty = random.randint(0, randomRangey) if shiftPositivey else random.randint(-randomRangey, 0)

    newPoint = (point[0] + randomShiftx, point[1] + randomShifty)
    if newPoint[0] < 0 or newPoint[0] >= boundaries[0] or newPoint[1] < 0 or newPoint[1] >= boundaries[1]:
        return point
    return newPoint

if __name__ == "__main__":
    main()