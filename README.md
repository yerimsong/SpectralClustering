# Locating Earthquake Hotspots Using Spectral Clustering

For the final project of Matrices and Linear Algebra (21-241), my partner and I implemented spectral clustering on both the k-nearest neighbor and e-neighborhood similarity graphs.

In short, for the k-nearest-neighbor similarity graph, vertices were connected with weighted edges based on how close they were. Meanwhile, the e-neighborhood graph had vertices connected by an edge if the distance between them was less than a specified e.

After creating these two graphs, we clustered data points into groups using linear algebra concepts. In particular, we derived a Laplachian matrix,
essentially a matrix representation of a graph, and then decomposed it into eigenvalues and eigenvectors.

This algorithm for spectral clustering was applied on real world data of earthquake locations from 1000 seismic events of MB > 4.0 near Fiji since
1964. The results uncovered the coordinates of earthquake hotspots.

Note: all code is written in Julia

A complete write-up can be found here:
[21241_Final_Project.pdf](https://github.com/yerimsong/SpectralClustering/files/9629650/21241_Final_Project.pdf)
