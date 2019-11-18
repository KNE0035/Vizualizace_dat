import numpy as np
import math
import cv2
from os import listdir
import random

spaceSize = (250, 250)

N = 4
E = 4
k = math.sqrt((spaceSize[0] * spaceSize[1]) / N)

class Vector2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if not isinstance(other, Vector2d):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def dist(self, other):
        minusPoint = self.substract(other)
        return minusPoint.nor

    def norm(self):
        return math.sqrt(math.pow(self.x, 2) + math.pow(self.y, 2))

    def add(self, other):
        return Vector2d(self.x + other.x, self.y + other.y)

    def substract(self, other):
        return Vector2d(self.x - other.x, self.y - other.y)

    def scalarMultiply(self, scalar):
        return Vector2d(self.x * scalar, self.y * scalar)

class ForcedPoint2d:
    def __init__(self, x, y):
        self.point = Vector2d(x, y)
        self.force = Vector2d(0, 0)

    def addToForce(self, force):
        self.force = self.force.add(force)


def main():
    pointsEdges = generateGraph(N);
    points = pointsEdges[0]
    edges = pointsEdges[1]

    t = 1
    iter = 20
    dt = 0.2
    drawGraph(points, edges, 0)
    for it in range(iter):
        fi = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    points[i].force = points[i].force.add(Fr(points[i].point, points[j].point))

        for edge in edges:
            firstPoint = points[edge[0]]
            secondPoint = points[edge[1]]
            firstPoint.force = firstPoint.force.substract(Fa(firstPoint.point, secondPoint.point))
            secondPoint.force = secondPoint.force.add(Fa(firstPoint.point, secondPoint.point))

        for forcePoint in points:
            forcePoint.point = forcePoint.point.add(forcePoint.force.scalarMultiply(1 / forcePoint.force.norm()).scalarMultiply(t * forcePoint.force.norm()))
            forcePoint.point.x = int(forcePoint.point.x)
            forcePoint.point.y = int(forcePoint.point.y)
            drawGraph(points, edges, 10)

    drawGraph(points, edges, 0)

def Fr(p1, p2):
    return p2.substract(p1).scalarMultiply(- (k * k) / math.pow(p1.substract(p2).norm(), 2))

def Fa(p1, p2):
    return p2.substract(p1).scalarMultiply(p1.substract(p2).norm() / k)

def generateGraph(nPoints):
    points = []
    edges = []
    offset = 10

    for i in range(nPoints):

        while True:
            randomX = random.randint(offset, spaceSize[0] - offset)
            randomY = random.randint(offset, spaceSize[1] - offset)
            point = ForcedPoint2d(randomX, randomY)

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

def drawGraph(points, edges, waitms):
    graphImg = np.zeros(spaceSize, dtype="uint8")

    for forcePoint in points:
        cv2.circle(graphImg, (forcePoint.point.x, forcePoint.point.y), 1, 255, -1)

    for edge in edges:
        cv2.line(graphImg, (points[edge[0]].point.x, points[edge[0]].point.y), (points[edge[1]].point.x, points[edge[1]].point.y), 100)

    cv2.imshow("Not procesed graph", graphImg)
    cv2.waitKey(waitms)

if __name__ == "__main__":
    main()