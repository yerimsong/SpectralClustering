#Package Imports:
using Pkg
# Pkg.add("ScikitLearn")
# Pkg.add("NearestNeighbors")
# Pkg.add("DataFrames")
# Pkg.add("Clustering")
# Pkg.add("RDatasets")

using Plots
using LinearAlgebra
using NearestNeighbors
using DataFrames
using Clustering
using RDatasets
using ScikitLearn
@sk_import datasets: make_circles

#Computes Laplacian using KNN similarity:
function LaplacianKNN(G, k, returnD::Bool=false) 

  W = zeros(size(G, 2), size(G, 2))

  for x = 1:size(G,2)

    bruteTree = BruteTree(G)
    point = G[:, x]
  
    #generate knn
    indexes, distances = knn(bruteTree, point, k+1, true)
  
    splice!(indexes, 1)
    splice!(distances, 1)
  
    #use knn to generate W
    for y = 1:k
      W[CartesianIndex(x,indexes[y])] = distances[y]
      W[CartesianIndex(indexes[y],x)] = distances[y]
    end
  
  end
  
  #generate D
  D = zeros(size(G, 2), size(G, 2))
  
  for x = 1:size(G,2)
    s = sum(W[x,:])
    D[CartesianIndex(x,x)] = s
  end
  
  #generate L
  L=D-W

  if(returnD)
    return L, D
  end

  return L
end

#Computes Laplacian using epsilon-neighbor similarity:
function LaplacianEpsilon(G, epsilon, returnD::Bool=false)
  
  W = zeros(size(G, 2), size(G, 2))

  for x = 1:size(G,2)
    point = G[:, x]

    for y=1:size(G,2)
      testPoint = G[:, y]

      #generate W using epsilon
      if x != y
        distance = sqrt(dot(point-testPoint,point-testPoint))

        if(distance < epsilon) 
          W[CartesianIndex(x,y)] = 1
          W[CartesianIndex(y,x)] = 1
        end
      end
    end
  end

  #generate D
  D = zeros(size(G, 2), size(G, 2))
  
  for x = 1:size(G,2)
    s = sum(W[x,:])
    D[CartesianIndex(x,x)] = s
  end
  
  #generate L
  L=D-W

  if(returnD)
    return L, D
  end

  return L
end

#Converts Laplacian to symmetric normalized:
function Lsym(L, D) 
  return D^(-1/2)*L*D^(-1/2)
end

#Converts Laplacian to random walk normalized:
function Lrw(L, D)
  return D^(-1)*L
end

#Computes clusters:
function clusterAssignments(L, k)

  V = real(eigvecs(L))
  V = transpose(V[:, 1:k])
  
  #run kmeans on the first k rows of L
  R = kmeans(V, k)
  a = assignments(R)
  
  return a
end

#Returns Laplacian given dataset:
@enum Datasource testdata irisdata clusterdata quakesdata mapsdata radardata circlesdata 
function getL(source::Datasource, k, epsilon)
  if(source==testdata)
    clusterSize = 20

    data1 = rand(1:clusterSize,clusterSize,1)
    data2 = rand(clusterSize+1:2*clusterSize,clusterSize,1)
    data3 = rand(1:clusterSize,clusterSize,1)

    data41 = hcat(1:clusterSize,data1)
    data42 = hcat(clusterSize+1:2*clusterSize,data2)
    data43 = hcat(2*clusterSize+1:3*clusterSize,data3)
    data4 = transpose(vcat(data41,data42, data43))

    LK0, DK0 = LaplacianKNN(data4, k, true) #10
    LE0, DE0 = LaplacianEpsilon(data4, epsilon, true) #10

    return data4, LK0, DK0, LE0, DE0
  elseif(source==irisdata)
    iris = dataset("datasets", "iris")
    features = collect(Matrix(iris[:, 1:4])')

    LK1, DK1 = LaplacianKNN(features, k, true) #10
    LE1, DE0 = LaplacianEpsilon(features, epsilon, true) #1

    return iris, LK1, DK1, LE1, DE0
  elseif(source==clusterdata)
    cluster = dataset("cluster", "xclara")
    features2 = collect(Matrix(cluster[:, 1:2])')
    
    LK2, DK2 = LaplacianKNN(features2, k, true) #10
    LE2, DE2 = LaplacianEpsilon(features2, epsilon, true) #20

    return cluster, LK2, DK2, LE2, DE2
  elseif(source==quakesdata)
    quakes = dataset("datasets", "quakes")
    features3 = transpose(collect(Matrix(quakes[:, 1:2])))
    
    LK3, DK3 = LaplacianKNN(features3, k, true) #15
    LE3, DE3 = LaplacianEpsilon(features3, epsilon, true) #5

    return quakes, LK3, DK3, LE3, DE3
  elseif(source==mapsdata)
    maps = dataset("HistData", "OldMaps")
    features4 = transpose(collect(Matrix(maps[:, 5:6])))
    
    LK4, DK4 = LaplacianKNN(features4, k, true) #10
    LE4, DE4 = LaplacianEpsilon(features4, epsilon, true) #10

    return maps, LK4, DK4, LE4, DE4
  elseif(source==radardata)
    radar = dataset("robustbase", "radarImage")
    features5 = transpose(collect(Matrix(radar[:, 1:2])))
    
    LK5, DK5 = LaplacianKNN(features5, k, true) #5
    LE5, DE5 = LaplacianEpsilon(features5, epsilon, true) #2
    
    return radar, LK5, DK5, LE5, DE5
  elseif(source==circlesdata)
    circles = make_circles(n_samples = 500, shuffle=true, noise=.1, factor=.3)
    
    circlesData = transpose(circles[1])

    LK6, DK6 = LaplacianKNN(circlesData, k, true) #15
    LE6, DE6 = LaplacianEpsilon(circlesData, epsilon, true) #.3

    return circlesData, LK6, DK6, LE6, DE6
  end
end

#Returns scatterplot given dataset and parameters:
@enum returnType KNN EPSILON

function getScatterPlot(source::Datasource, type::returnType, title, clustervar, numClusters)
  k = 1
  epsilon = 1

  #run if knn
  if(type==KNN)
    k=clustervar
    
  #run if epsilon
  elseif(type==EPSILON)
    epsilon = clustervar
  end

  #get Laplacian
  data = getL(source, k, epsilon)
  plotData = data[1]

  if(type==KNN)
    L=data[2]
  elseif(type==EPSILON)
    L=data[4]
  end

  #return plot from Laplacian
  if(source==testdata)
    return scatter(plotData[1, :], plotData[2, :],marker_z=clusterAssignments(L, numClusters), legend = false, title=title)
  elseif(source==irisdata)
    return scatter(plotData.PetalLength, plotData.PetalWidth, marker_z = clusterAssignments(L, numClusters), color =:blue, legend = false, title=title)
  elseif(source==clusterdata)
    return scatter(plotData.V1, plotData.V2, marker_z = clusterAssignments(L, numClusters), color =:blue, legend = false, title=title) #2
  elseif(source==quakesdata)
    return scatter(plotData.Lat, plotData.Long, marker_z = clusterAssignments(L, numClusters), color =:blue, legend = false, title=title) #3
  elseif(source==mapsdata)
    return scatter(plotData.Lat, plotData.Long, marker_z = clusterAssignments(L, numClusters), color =:blue, legend = false, title=title) #4
  elseif(source==radardata)
    return scatter(plotData.XCoord, plotData.YCoord, marker_z = clusterAssignments(L, numClusters), color =:blue, legend = false, title=title) #5
  elseif(source==circlesdata)
    return scatter(plotData[1,:], plotData[2,:], marker_z = clusterAssignments(L, numClusters), color =:blue, legend = false, title=title) #6
  end
end

#Generate graphs:
gr()

display(getScatterPlot(testdata, KNN, "Test Data (k=10)", 10, 3))

display(getScatterPlot(irisdata, KNN, "Iris (k=10)", 10, 3))

display(getScatterPlot(clusterdata, EPSILON, "Cluster Test Data ($\epsilon$=20)", 20, 3))

display(getScatterPlot(quakesdata, KNN, "Earthquake Locations (k=20)", 20, 3))

display(getScatterPlot(mapsdata, KNN, "test", 10, 2))

display(getScatterPlot(radardata, KNN, "Radar Survey Locations (k=5)", 5, 3))

display(getScatterPlot(circlesdata, EPSILON, "Circle Test Data ($\epsilon$=.3)", .3, 2))



# RUNNING TESTS

#TESTING DIFFERENT VALUES FOR EPSILON AND K
radarE1 = getScatterPlot(radardata, EPSILON, "$\epsilon$=1", 1, 3)
radarE2 = getScatterPlot(radardata, EPSILON, "$\epsilon$=2", 2, 3)
radarE3 = getScatterPlot(radardata, EPSILON, "$\epsilon$=3", 3, 3)
radarE4 = getScatterPlot(radardata, EPSILON, "$\epsilon$=4", 4, 3)

radarK1 = getScatterPlot(radardata, KNN, "k=2", 2, 3)
radarK2 = getScatterPlot(radardata, KNN, "k=5", 5, 3)
radarK3 = getScatterPlot(radardata, KNN, "k=10", 10, 3)
radarK4 = getScatterPlot(radardata, KNN, "k=20", 20, 3)

display(plot(radarE1, radarE2, radarE3, radarE4, layout=(2,2), legend=false))
display(plot(radarK1, radarK2, radarK3, radarK4, layout=(2,2), legend=false))

circlesE1 = getScatterPlot(circlesdata, EPSILON, "$\epsilon$=.1", .1, 2)
circlesE2 = getScatterPlot(circlesdata, EPSILON, "$\epsilon$=.2", .2, 2)
circlesE3 = getScatterPlot(circlesdata, EPSILON, "$\epsilon$=.5", .5, 2)
circlesE4 = getScatterPlot(circlesdata, EPSILON, "$\epsilon$=1", 1, 2)

circlesK1 = getScatterPlot(circlesdata, KNN, "k=2", 2, 2)
circlesK2 = getScatterPlot(circlesdata, KNN, "k=5", 5, 2)
circlesK3 = getScatterPlot(circlesdata, KNN, "k=10", 10, 2)
circlesK4 = getScatterPlot(circlesdata, KNN, "k=50", 50, 2)

display(plot(circlesE1, circlesE2, circlesE3, circlesE4, layout=(2,2), legend=false))
display(plot(circlesK1, circlesK2, circlesK3, circlesK4, layout=(2,2), legend=false))

# SPECTRAL VS K-MEANS
circlesE = getScatterPlot(circlesdata, EPSILON, "Spectral: $\epsilon$=.5", .5, 2)
circlesK = getScatterPlot(circlesdata, EPSILON, "Spectral: k=15", 15, 2)

circlesKM = make_circles(n_samples = 500, shuffle=true, noise=.1, factor=.3)
circlesData = transpose(circlesKM[1])

#run k-means on raw circle dataset without clustering
R = kmeans(circlesData, 2)
a = assignments(R)

circlesKMPlot = scatter(circlesData[1,:], circlesData[2,:], marker_z = a, color =:blue, legend = false, title="K Means")

display(circlesKMPlot)
