# Clustering-Using-KMeans

[K-means](https://github.com/Surya-Murali/Clustering-Using-KMeans/blob/master/KMeans.py) is an unsupervised clustering algorithm that tries to partition a set of points into *k* clusters. It is used when you have to group a collection of stuff into various clusters.

### The Algorithm:

* Assign random positions for *k* centroids
* Compute the distance of each point from the centroids and assign each point to its nearest centroid, thereby forming *k* clusters
* Take the mean of the distance of the points assigned to each centroid. This now becomes the positions of the new centroids
* Now check the error(distance) between the positions of old and new centroids. 
* If the error is not equal to 0, repeat steps 3 and 4. If the positions of the old and new centroids match, then the required clusters are formed!

This [gif](https://github.com/Surya-Murali/Clustering-Using-KMeans/blob/master/KMeans.gif) might help you better understand the algorithm
![alt text](https://github.com/Surya-Murali/Clustering-Using-KMeans/blob/master/KMeans.gif)

However, using Python's [Scikit](https://github.com/Surya-Murali/Clustering-Using-KMeans/blob/master/KMeansUsingScikit.py) to perform KMeans is much simpler
The outputs can be found [here.](https://github.com/Surya-Murali/Clustering-Using-KMeans/tree/master/Output)


### Some practical applications:

* Pricing Segmentation
* Customer Need Segmentation
* Loyalty Segmentation
* Where do millionaires live?
* Create stereotypes from demographics data
