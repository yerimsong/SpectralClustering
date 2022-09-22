# Locating Earthquake Hotspots Using Spectral Clustering

For the final project of Matrices and Linear Algebra (21-241), my partner and I implemented the spectral clustering algorithm in two different ways.

The first method used a k-nearest-neighbor similarity graph. It involved connecting vertices with weighted edges based on how close they were.
The second method, e-neighborhood graph, was derived similarly. However, vertices were connected by an edge if the distance between them was less than the
specified e.
After creating these two graphs, we clustered data points into groups using linear algebra concepts. In particular, we derived a Laplachian matrix,
essentially a matrix representation of a graph, and then decomposed it into eigenvalues and eigenvectors.

A complete write-up can be found here.
[21241_Final_Project.pdf](https://github.com/yerimsong/SpectralClustering/files/9629650/21241_Final_Project.pdf)
