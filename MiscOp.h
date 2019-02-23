//
// Created by brahmpowell on 7/15/16.
//

#ifndef FIRSTPROJWITHEIGEN_MISCOP_H
#define FIRSTPROJWITHEIGEN_MISCOP_H

#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;

class MiscOp
{
public:
    static MatrixXd MultElByEl(MatrixXd A, MatrixXd B);
    //static MatrixXd AddElByEl
    static void DimCheck(MatrixXd A, MatrixXd B);
};


#endif //FIRSTPROJWITHEIGEN_MISCOP_H
