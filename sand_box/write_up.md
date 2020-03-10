## Introduction 

## Method 
### Null Distribution
Three methods were chosen to create null distributions from the data
1. __Shuffle the data__
takes all the original data points and shuffles the order in the
variables to remove correlation between the variables

2. __Max Min Uniform Distribution__
is generated from a uniform distribution between the minumum and
maximum values of the variable

3. __PCA Distribution__
takes the eigan vectures of the data set are gained through PCA, these are
then used to transform a random data set generated from a  single
gaussian distribution. The resulting data set is one with only one
cluster yet maintains the relationships between the variables

To create a null distribution, 500 test data sets were generated

### Cluster Seperation Metrics
Three Seperation metrics are used:

1. __Huberts Gamma Statistic__
is a measure of how much the high distances
between variables correlates with cluster membership. It uses 2 matrices
the distance matrix which was the basis of clustering (D) and a matrix
recording cluster membership  where the value at point (i,j) is 1 if
they are from different clusters and 0 if they are from the same.
The statistic = the sum of D(i,j) * C(i,j) for i in 1-n and j in 2 -n /
the number of point pairs. The higher the value the better cluster
structure

2. __Normalised Gamma Statistic__
is the normalised version of the statistic above. The statistic =
(the sum of D(i,j)-mean(D) * C(i,j)- mean(C) for i in 1-n and j in 2 -n /
the number of point pairs)/var(D)*var(C). This returns a value between
0 and 1 with high being more clustered

3. __Total Within Cluster Sum of Squares__
is the sum of the distances from each point to its assigned cluster
center, the smaller the distance the better.

### Cluster Methodology
We apply k-means to each data set using a k++ initialisation with
50 resamples which then returns the optimum result


### Test Data Generation
The data was generated using SciKit learn Make Classifications function
from the datasets module. 4 parameters of the data are altered we used
a full factorial experimental design:

1. Number of clusters - 2,4,5
2. Number of features - 10,20
3. % Noise features - 0% 10% 50%
4. Seperation (measured in size of hypercube between clusters) - 0.5,1,3

This resulted in 54 distinct data sets.

### Overall Experiment Structure

1. 54 Datasets were created
2. For each data set 500 null distributions were made with each null
distribution method (total 1500 null distributions
3. K-means was run on the original data set and 1500 null datasets
and the three cluster seperation metrics were returned, for k = 2-6
4. The mean and standard deviation is returned for each null distribution
method, for each seperation metric and for each cluster number
5. The seperation metric score and p value for the original data set
is returned for each null distribution
method, for each seperation metric and for each cluster number

### Experiment Outcomes

Each distribution method and Cluster metric will be compared on two
different fronts:

1. the size of the distributions of each metric
2. the accuracy of each method and distribution paring



