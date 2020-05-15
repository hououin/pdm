import pickle
import numpy as np
from numpy import linalg as LA
import sys
import random
import math
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import time


DDEPTH = 1
NUM_LANDMARKS = 50
NUM_X = 6
NUM_LAYER = 15
np.set_printoptions(threshold=sys.maxsize)

timer1 = time.time()

x = []
for i in range(NUM_X):
    with open(f"data/landmarks/x{i}.pickle", "rb+") as f:
        x.append(np.asarray(pickle.load(f)).flatten())



avg = np.mean(x,axis=0)


cov_matrix = np.outer((x[0]-avg),(x[0]-avg))

for i in range(1,NUM_X):
    cov_matrix += np.outer((x[i]-avg),(x[i]-avg))

cov_matrix = cov_matrix / (NUM_X-1)

eigen_values, eigen_vectors = LA.eigh(cov_matrix)

b_eigen_value = eigen_values[-1]

b_eigen_vector = eigen_vectors[:,-1]

random.seed(10)
a = -3*math.sqrt(b_eigen_value)
b = 3*math.sqrt(b_eigen_value)
weight = random.uniform(a,b)

x_crtica = avg + weight * b_eigen_vector

# for i in range(0, len(x_crtica)-2, 3):
#     print(round(x_crtica[i], 0), round(x_crtica[i+1],0), round(x_crtica[i+2],0), "###", avg[i], avg[i+1], avg[i+2])

x_crtica = np.around(x_crtica)
# print(len(x_crtica))
# print(x_crtica)


x_data = []
y_data = []
z_data = []



for i in range(0,len(x_crtica),3):
    x_data.append(x_crtica[i])
    y_data.append(x_crtica[i+1])
    z_data.append(x_crtica[i+2])



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x_data, y_data, z_data, c=z_data, cmap="Greens")
fig.show()

###############################################################################

def izpisArrray(a):
    print(a)



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




positions_array = []
layer_position_array = []
poly_array = []
inx = 0



for i in range(0,len(x_crtica),3):
    if inx == NUM_LANDMARKS or i == (len(x_crtica)-3):
        #print(layer_position_array)
        origin = layer_position_array[0]
        sorted_layer_position_array = sorted(layer_position_array, key=clockwiseangle_and_distance)
        #print(sorted_layer_position_array)
        poly = Polygon(sorted_layer_position_array)
        poly_array.append(poly)
        layer_position_array = []
        inx = 0

    inx += 1
    t = [int(x_crtica[i]),int(x_crtica[i+1]),int(x_crtica[i+2])]
    positions_array.append(t)
    layer_position_array.append(t[0:2])


# p1 = Point(180,160)
# print(p1.within(poly_array[0]))
print(len(poly_array))
#print(len(positions_array))



np_izhod = np.zeros((500,500,80))

for i in range(len(positions_array)):
    p_z = int(positions_array[i][2])
    p_y = int(positions_array[i][1])
    p_x = int(positions_array[i][0])
    np_izhod[p_x][p_y][p_z] = 128




# for i in range(80):
#     print(f"layer{i}")
#     for j in range(500):
#         for k in range(500):
#             if i % DDEPTH == 0 and i != 0:
#                 inx_poly = int(i/5 - 1)
#                 if np_izhod[k][j][i]!=128:
#                     p = Point(j, k)
#                     if p.within(poly_array[inx_poly]):
#                         #print(k,j,i)
#                         np_izhod[k][j][i] = 255

for i in range(15):
    print(f"layer{i}")
    for j in range(500):
        for k in range(500):
                if np_izhod[k][j][i]!=128:
                    p = Point(j, k)
                    if p.within(poly_array[i]):
                        #print(k,j,i)
                        np_izhod[k][j][i] = 255





# print(np_izhod.shape)
#
# izhod = np_izhod.flatten()

# counter1 = 0
# counter2 = 0
# for i in range(len(izhod)):
#     if izhod[i] == 255:
#         counter2+=1
#     if izhod[i] == 128:
#         counter1+=1
#
#print(counter1,counter2)

# result1 = np.where(izhod == 128)
# print(result1)
# print(len(result1))


#binaryArray = bytearray(izhod)


with open("izhod.raw","wb+") as f:
    for i in range(80):
        for j in range(500):
            for k in range(500):
                f.write(bytearray([int(np_izhod[k][j][i])]))

f.close()



print(time.time() - timer1)


print("Process finished")



