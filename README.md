# Clustering-Using-KMeans

K-means is an unsupervised clustering algorithm that tries to partition a set of points into *k* clusters. It is used when you have to group a collection of stuff into various clusters.

### The Algorithm:

* Assign random positions of *k* centroids
* Compute the distance of each point from the centroids and assign each point to its nearest centroid, thereby forming *k* clusters
* Take the mean of the distance of the points assigned to each centroid. This now becomes the positions of the new centroids
* Now check the error(distance) between the positions of old and new centroids. 
* If the error is not equal to 0, repeat steps 3 and 4. If the positions of the old and new centroids are the same, that becomes our final output!

This [gif](https://github.com/Surya-Murali/Clustering-Using-KMeans/blob/master/KMeans.gif) might help you better understand the KMeans algorithm

![alt text](https://github.com/Surya-Murali/Clustering-Using-KMeans/blob/master/KMeans.gif)

However, using Python's Scikit to perform KMeans is much easier
The outputs can be found [here.](https://github.com/Surya-Murali/Clustering-Using-KMeans/tree/master/Output)


### Some practical applications:

* Pricing Segmentation
* Customer Need Segmentation
* Loyalty Segmentation
* Where do millionaires live?
* Create stereotypes from demographics data
