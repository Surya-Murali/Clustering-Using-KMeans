from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import pandas as pd

#Reading data
data = pd.read_csv('C:/Users/surya/Desktop/DecemberBreak/RunPy/KMeans/Dataset/clustering.csv')
print("Data Shape: ", data.shape)
print("Data Head: ", data.head())

#Assigning 3 clusters to the data
print("Number of Clusters = 3")
model = KMeans(n_clusters=3)

#Scaling data to normalise it for better results
#Fitting the model with the scaled data
model = model.fit(scale(data))

#Model labels of the clusters
print("Labels of the Clusters: ",model.labels_)

#Visulaising it
plt.figure(figsize=(8, 6))

#Printing the columns of the data
print("Column values of the data: ", data.values[:,0])
#Printing the rows of the data
print("Row values of the data: ", data.values[:,1])

#Since 'data' is a dataframe, data.values is used
#Assigning the labels of the clusters to different colors
plt.scatter(data.values[:,0], data.values[:,1], c=model.labels_.astype(float), edgeColor= 'k')
plt.show()