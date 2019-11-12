import numpy as np
import math
import cv2
from os import listdir
import random

spaceSize = (250, 250)

def main():
    pointsEdges = generateGraph(4);
    points = pointsEdges[0]
    edges = pointsEdges[1]
    drawGraph(points, edges)

def generateGraph(nPoints):
    points = []
    edges = []
    offset = 10

    for i in range(nPoints):

        while True:
            randomX = random.randint(offset, spaceSize[0] - offset)
            randomY = random.randint(offset, spaceSize[1] - offset)
            point = (randomX, randomY)

            if not point in points:
                points.append(point)
                break

    for i in range(nPoints):
        randomPointIndex = 0

        while True:
            randomPointIndex = random.randint(0, nPoints - 1)
            if i != randomPointIndex:
                edges.append((i, randomPointIndex))
                break

    return (points, edges)

def drawGraph(points, edges):
    graphImg = np.zeros(spaceSize, dtype="uint8")

    print(len(points))
    print(len(edges))

    for point in points:
        cv2.circle(graphImg, point, 1, 255, -1)

    for edge in edges:
        cv2.line(graphImg, points[edge[0]], points[edge[1]], 100)

    cv2.imshow("Not procesed graph", graphImg)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()