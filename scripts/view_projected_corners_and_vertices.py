import os
import cv2
import numpy

dir_ = os.path.dirname(os.path.abspath(__file__))

imagesdir   = os.path.join(dir_, '../render/images')
cornersdir  = os.path.join(dir_, '../render/corners')
verticesdir = os.path.join(dir_, '../render/vertices')
files_      = os.listdir(imagesdir)

for name in files_:
    image    = cv2.imread(os.path.join(imagesdir, name))
    H, W, _  = image.shape

    corners  = numpy.loadtxt(os.path.join(cornersdir,  name.replace('png', 'txt')))
    vertices = numpy.loadtxt(os.path.join(verticesdir, name.replace('png', 'txt')))

    corners_image  = image.copy()
    vertices_image = image.copy()

    for point in corners:
        if point[0] <= 1. and point[1] <= 1. and point[0] > 0 and point[1] > 0:
            corners_image = cv2.circle(corners_image, (int(point[0]*W), int(point[1]*H)), 1, (0,0,255), -1)

    for point in vertices:
        if point[0] <= 1. and point[1] <= 1. and point[0] > 0 and point[1] > 0:
            vertices_image = cv2.circle(corners_image, (int(point[0]*W), int(point[1]*H)), 1, (0,0,255), -1)


    cv2.imshow('Original', image)
    cv2.imshow('Corners',  corners_image)
    cv2.imshow('Vertices', vertices_image)
    cv2.waitKey(0)


