from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

data = pd.read_csv('C:/Users/surya/Desktop/DecemberBreak/RunPy/KMeans/clustering.csv')

input1 = data['V1'].values
input2 = data['V2'].values
inputArray = np.array(list(zip(input1, input2)))

plt.scatter(input1, input2, c='black', s=5)

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
    
k = 3
C_x = np.random.randint(0, np.max(inputArray)-20, size=k)
C_y = np.random.randint(0, np.max(inputArray)-20, size=k)

C = np.array(list(zip(C_x, C_y)), dtype=np.float32)

plt.scatter(input1, input2, c='black', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')

C_old = np.zeros(C.shape)

clusters = np.zeros(len(inputArray))

error = dist(C, C_old, None)

while error != 0:
    for i in range(len(inputArray)):
        distances = dist(inputArray[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    for i in range(k):
        points = [inputArray[j] for j in range(len(inputArray)) if (clusters[j] == i)]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
    
    #While Loop ends
    colors = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots()
    for i in range(k):
            points = np.array([inputArray[j] for j in range(len(inputArray)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=13, c=colors[i], marker = 'o')
  
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='yellow')
    plt.show()