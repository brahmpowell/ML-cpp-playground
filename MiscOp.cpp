//
// Created by brahmpowell on 7/15/16.
//

#include "MiscOp.h"
using namespace std;

MatrixXd MiscOp::MultElByEl(MatrixXd A, MatrixXd B)
{
    DimCheck(A,B);

    long rows = A.rows();
    long cols = A.cols();
    MatrixXd C(rows, cols);

    for (long i = 0; i < rows; i++)
    {
        for (long j = 0; j < cols; j++)
        {
            C(i,j) = A(i,j)*B(i,j);
        }
    }

    return C;
}

void MiscOp::DimCheck(MatrixXd A, MatrixXd B)
{
    // Make sure the dimensions of A and B are the same
    try
    {
        if (A.rows() != B.rows() && A.cols() != B.cols())
        {
            throw "WOAH! When doing matrix element-by-element operations, make sure they are the same size!";
        }
        else if (A.rows() != B.rows())
        {
            throw "WOAH! When doing matrix element-by-element operations, make sure they have the same number of rows!";
        }
        else if (A.cols() != B.cols())
        {
            throw "WOAH! When doing matrix element-by-element operations, make sure they have the same number of cols!";
        }
    }
    catch(string e)
    {
        cout << e << endl;
    }
    return;
}



