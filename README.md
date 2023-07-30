# Semi-Convex Hull Tree

A data structure optimized for querying the exact K nearest neighbors (KNN) of a set of points.
Implementation of the paper [Semi-Convex Hull Tree: Fast Nearest Neighbor Queries for Large Scale Data on GPUs](https://ieeexplore.ieee.org/document/8594919).
[Official implementation repository](https://github.com/XFastDataLab/Semi-convex_Hull_Tree).

## Dependencies

-   [Eigen3](https://eigen.tuxfamily.org)
-   OpenMP (optional)

## Install

Just copy `schtree.h` to your project. Must be available
Eigen3 library. To install it you can follow [this guide](https://eigen.tuxfamily.org/dox/GettingStarted.html).
Optionally you can activate OpenMP using your compiler.

## Example

~~~cpp
#include <iostream>
#include <vector>
#include <schtree.h>

const int D = 5; // number of dimensions
const int N = 1000; // number of points

int main() {
     std::vector<sch::Vec<D>> points(N); // set of points
     sch::Tree<D> tree(points); // the data structure
     sch::Vec<D> query; // the point to search for its k nearest neighbors
     auto res = tree.knnSearch(query, 30); // find its 30 nearest neighbors
     for (const auto& i : res) {
         std::cout << points[i.idx] << "\n"; // print the results
     }
}
~~~

## Test

To run the tests use `make test`. This will compile the `test/test.cpp` file
and will run it using the dataset located in `test/dataset.csv`, which corresponds to
to [Dataset of songs in Spotify](https://www.kaggle.com/datasets/mrmorj/dataset-of-songs-in-spotify).

**Note:** To run the tests you will probably have to modify `Makefile`.

## Documentation

### namespace sch

Where are all the functions and classes of the tree

### sch::Vec<DIM, Scalar = float>

It is an alias of an Eigen3 matrix of DIM rows and 1 column. By default `Scalar` is `float`.

### sch::KnnResult<Scalar = float>

It is the result of the KNN search. To access individual points use:

~~~cpp
for (const auto& i : res) {
     i.idx; // the index of the point
     i.dist; // the distance from the point to q
}
~~~

### sch::Tree<DIM, Scalar = float>

The data structure. Aliases in the class:

- `vec`: `Vec<DIM, Scalar>`.
- `knn_result`: `KnnResult<Scalar>`.

### sch::Tree(std::vector<vec>& points, bool copy = false)

- `points`: The points to build the tree.
- `copy`: If it is true it copies the points in the tree, if it is false it only uses a pointer to the points.

### knn_result knnSearch(const vec& q, int k, bool sort = false)

- `q`: the point to be consulted by its nearest neighbors.
- `k`: the number of neighbors that we want to return.
- `sort`: If true, sort the points in the result by the distances ascending,
     if it is false, leave the points unordered.

### void knnBulkSearch(std::vector<knn_result>& query_results, const std::vector<vec>& query_points, int k, bool sort = false)

- `query_results`: Where the query results are stored.
- `query_points`: The points to query for its nearest neighbors.
- `k`: The number of neighbors that we want to return for each query.
- `sort`: If true, sort the points in the results by the distances ascending,
     if it is false, leave the points unordered.
- **Note:** `knnSearch` is independent and thread-safe, so `knnBulkSearch`
     is parallelized with OpenMP. It consumes a lot of memory as it saves
     all results.

