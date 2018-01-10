from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

#Reading data
data = pd.read_csv('C:/Users/surya/Desktop/DecemberBreak/RunPy/KMeans/Dataset/clustering.csv')
print("Data Shape: ", data.shape)
print("Data Head: ", data.head())

# Getting the values and plotting it
input1 = data['V1'].values
input2 = data['V2'].values
inputArray = np.array(list(zip(input1, input2)))
print("Input Array: ", inputArray[1:6])

#A Scatterplot in black color and size 5 with the 2 input columns 
plt.scatter(input1, input2, c='black', s=5)
#plt.show()

#Euclidean Distance calculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
    
print("Example of dist(1, 2, None): ", dist(1, 2, None))

print("Max: ", np.max(inputArray))
print("Min: ", np.min(inputArray))

# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(inputArray)-20, size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(inputArray)-20, size=k)

print("X Coordinate of k centroids: ", C_x)
print("Y Coordinate of k centroids: ", C_y)

#X & Y coordinate of 3 centroids zipped together
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print("k Centroids (new): \n", C)

# Plotting the Centroids as Stars
plt.scatter(input1, input2, c='black', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
#plt.show()

# To store the value of centroids when it updates
print ("C.shape: ", C.shape)
C_old = np.zeros(C.shape)
print ("C_old Zeros Centroid: \n", C_old)
# Cluster Lables(0, 1, 2)
print("Input Array Range: ", range(len(inputArray)))
print("Input Array Length: ", len(inputArray))
clusters = np.zeros(len(inputArray))
print ("Zeros Input Array: \n", clusters)

# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
print("Initial Error: ", error)
print("-------------------------------------------------")

# Loop will run till the error becomes zero
while error != 0:
    print("Error of While loop: ", error)
    
    #Compute the distance
    #Assign each distance to its closest cluster
    for i in range(len(inputArray)):
        distances = dist(inputArray[i], C)
        #print ("dist: ",distances)
        #dist:  [ 78.08083455  80.01009709  16.14851343]
        #dist:  [ 74.44720446  75.00936282  15.00144485]
        #dist:  [ 58.92797979  74.13522415   5.47219653]
        #dist:  [ 84.42182071  87.65757839  20.90804849]
        #dist:  [ 65.83390246  87.98558267  15.59599391]
        cluster = np.argmin(distances)
        #print("cluster: ", cluster)
        #cluster:  0
        #cluster:  2
        #cluster:  2
        #cluster:  1
        #cluster:  1
        #cluster:  2
        #cluster:  2
        #Clusters assigned
        clusters[i] = cluster
    
    # Storing the old centroid values
    C_old = deepcopy(C)
    print("Old centroid values: \n", C_old)
    
    # Finding the new centroids by taking the average value
    #i <- (0 to 3) j<- (0 to 3000) (3 & 3000 not included) k = 3
    #A 'for' loop with an 'if' filter
    #For loop runs 3 times here
    for i in range(k):
        points = [inputArray[j] for j in range(len(inputArray)) if (clusters[j] == i)]
        #print(i)
        #0
        #1
        #2
        #print("Points: ", points)
        #Points:  [array([ 2.072345, -3.241693]), array([ 17.93671,  15.78481]), array([ 1.083576,  7.319176]),....
        
        #axis = 0 indicates mean of each col; axis = 1 indicates mean of each row
        #Get the mean centroid for each value of k
        #New Centroid positions by taking the mean of the points assignedd to each cluster
        C[i] = np.mean(points, axis=0)
        #print("Mean points C[",i,"]: ",C[i])
        #Mean points C[ 0 ]:  [  9.47804546  10.68605232]
        #Mean points C[ 1 ]:  [ 69.92418671 -10.1196413 ]
        #Mean points C[ 2 ]:  [ 40.68362808  59.71589279]
    
    print("New centroid values: \n", C)
    error = dist(C, C_old, None)
    print("******** While Loop Ends ********")
    
    colors = ['r', 'g', 'b', 'y']
    #plt.subplots() is a function that returns a tuple containing a figure and axes object(s). Thus when using fig, ax = plt.subplots() you unpack this tuple into the variables fig and ax. Having fig is useful if you want to change figure-level attributes or save the figure as an image file later (e.g. with fig.savefig('yourfilename.png'). 
    fig, ax = plt.subplots()
    print("Fig: ", fig)
    print("ax: ", ax)
    
    for i in range(k):
            #For assigning colors! (color[i])        
            points = np.array([inputArray[j] for j in range(len(inputArray)) if clusters[j] == i])
            ax.scatter(points[:, 0], points[:, 1], s=13, c=colors[i], marker = 'o')
    #print("X axis Points: \n", points[:, 0])
    #print("Y axis Points: \n", points[:, 1])
    print("X axis Centroid: ", C[:, 0])
    print("X axis Centroid: ", C[:, 1])
    
    #A star marker for centroids       
    ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='yellow')
    plt.show()