//
// Created by brahmpowell on 7/11/16.
//

#ifndef FIRSTPROJWITHEIGEN_NN1_H
#define FIRSTPROJWITHEIGEN_NN1_H

#include <iostream>
#include <vector>
#include "Eigen/Dense"

using std::vector;
using Eigen::MatrixXd;

class NN1
{
public:
    NN1(int layers, int nPerLayer, int nInputFeatures, int nOutputs);
    void Train(MatrixXd x, MatrixXd y);
    MatrixXd Predict(MatrixXd x);
private:
    vector<MatrixXd> w;
    int NumInputFeatures;
    int NumOutputFeatures;
    int NumLayers;
    int ArbitraryHiddenLayerSize;
    bool IsTrained;
    static MatrixXd Sigmoid(MatrixXd input);
    static MatrixXd SigmoidBackGradient(MatrixXd input);
};


#endif //FIRSTPROJWITHEIGEN_NN1_H
