import numpy as np
import cv2
import czifile
import pickle
import matplotlib.pyplot as plt
import scipy.misc
import math
import random

DDEPTH = 1
NUM_LANDMARKS = 50




def cropImage(im_cell):
    sum_i = 0
    sum_j = 0
    stevec = 0
    for i in range(im_cell.shape[0]):
        for j in range(im_cell.shape[1]):
            if im_cell[i][j][0] != im_cell[i][j][1] != im_cell[i][j][2]:
                sum_j += j
                sum_i += i
                stevec += 1

    average_i = int(sum_i/stevec)
    average_j = int(sum_j/stevec)

    #cv2.circle(im_cell_1, (average_i, average_j), 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)

    new_i_z = average_i - 250
    new_j_z = average_j - 250

    crop_image = im_cell[new_i_z:new_i_z+500, new_j_z:new_j_z+500].copy()
    # print(crop_image.shape)
    # cv2.imshow("cropped", crop_img)
    # celica_edge = cv2.Canny(crop_img,100,200)
    # cv2.imshow('edge_detection1',celica_edge)
    return crop_image




def findEdge(crop_img):
    sigma = 25
    for i in range(500):
        for j in range(500):
            dif1 = abs(int(crop_img[i][j][0]) - int(crop_img[i][j][1]))
            dif2 = abs(int(crop_img[i][j][1]) - int(crop_img[i][j][2]))
            dif3 = abs(int(crop_img[i][j][0]) - int(crop_img[i][j][2]))
            if (dif1 < sigma or dif2 < sigma or dif3 < sigma) and crop_img[i][j][2] <= 200:
                crop_img[i][j][0] = 0
                crop_img[i][j][1] = 0
                crop_img[i][j][2] = 0
            elif crop_img[i][j][0] == crop_img[i][j][1] == crop_img[i][j][2]:
                crop_img[i][j][0] = 0
                crop_img[i][j][1] = 0
                crop_img[i][j][2] = 0
            else:
                crop_img[i][j][0] = 0
                crop_img[i][j][1] = 0
                crop_img[i][j][2] = 255

    # cv2.imshow("blackened", crop_img)
    celica_edge = cv2.Canny(crop_img,100,200)
    #cv2.imshow('edge_detection1',celica_edge)

    return celica_edge

#
# def distanceBetweenPoints(a,b):
#     return math.sqrt(math.pow(b[0]-a[0]) + math.pow(b[1]-a[0]))


origin = []
refvec = [0,1]


def clockwiseangle_and_distance(point):
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector





def getVectorOfEdge(celica_edge):
    counter = 0

    # temp_prev = 0
    # distance = 0
    landmarks = np.zeros((500, 500))
    vector_edge = []
    for i in range(500):
        for j in range(500):
            if celica_edge[i][j] == 255:
                vector_edge.append([i,j])
                counter+=1

    return vector_edge


def getVectorOfLandmakrs(sorted_vec, num_landmarks, depth):
    layer_landmarks = []
    n = len(sorted_vec)
    d = int(n/num_landmarks)
    counter = 0
    for i in range(0,n,d):
        if counter == num_landmarks:
            break
        else:
            layer_landmarks.append([sorted_vec[i][0],sorted_vec[i][1]])
            vec_landmarks.append([sorted_vec[i][0],sorted_vec[i][1], depth*DDEPTH])
            counter += 1

   # showLandmark(layer_landmarks)


def showLandmark(x):
    landmarks = np.zeros((500, 500))
    for i in range(len(x)):
        landmarks[int(x[i][0])][int(x[i][1])] = 255

    cv2.imshow("landmarks",landmarks)
    cv2.waitKey(0)

if __name__ == "__main__":



    depth = 1
    vec_landmarks = []
    for i in range(30,45):
        im_cell = cv2.imread(f"data/p16-1/p16-{i}.jpg")
        # print(im_cell)
        # cv2.imshow(f'p06-{i}.jpg',im_cell)
        crop_img = cropImage(im_cell)
        celica_edge = findEdge(crop_img)
        vec_edge = getVectorOfEdge(celica_edge)
        origin = vec_edge[0]
        #print(origin)
        sorted_vec_edge = sorted(vec_edge, key=clockwiseangle_and_distance)
        getVectorOfLandmakrs(sorted_vec_edge, NUM_LANDMARKS, depth)
        depth += 1
        print(len(vec_landmarks))




    with open("data/landmarks/x5.pickle","wb+") as f:
        pickle.dump(vec_landmarks, f)

    cv2.destroyAllWindows()
    print("Process finished")
