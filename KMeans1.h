//
// Created by brahmpowell on 7/20/16.
//

#ifndef FIRSTPROJWITHEIGEN_KMEANS1_H
#define FIRSTPROJWITHEIGEN_KMEANS1_H

#include <iostream>
#include <vector>
#include "Eigen/Dense"

using std::vector;
using Eigen::MatrixXd;
using Eigen::ArrayXXd;

class KMeans1
{
public:
    KMeans1();
    void Train(ArrayXXd X, int k);
    //int Predict();
private:
    ArrayXXd c;     // Centroids


};


#endif //FIRSTPROJWITHEIGEN_KMEANS1_H
