//
// Created by brahmpowell on 7/18/16.
//

#ifndef FIRSTPROJWITHEIGEN_NN2_H
#define FIRSTPROJWITHEIGEN_NN2_H

#include <iostream>
#include <vector>
#include "Eigen/Dense"

using std::vector;
using Eigen::MatrixXd;

class NN2
{
public:
    NN2(int layers, int nPerLayer, int nInputFeatures, int nOutputs);
    static NN2 EZTrain(MatrixXd X, MatrixXd Y, int layers, int nPerLayer);
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
    static MatrixXd BiasAdd(MatrixXd input);
    //static MatrixXd BiasRemove(MatrixXd input);
};


#endif //FIRSTPROJWITHEIGEN_NN2_H
