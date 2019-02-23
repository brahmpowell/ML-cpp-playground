//
// Created by brahmpowell on 7/18/16.
//

#include "NN2.h"
#include "MiscOp.h"
#include <tgmath.h>

using namespace std;
using namespace Eigen;

NN2::NN2(int layers,
         int nPerLayer,
         int nInputFeatures,
         int nOutputs) :
        w(vector<MatrixXd>(layers)),
        NumInputFeatures(nInputFeatures),
        NumOutputFeatures(nOutputs),
        NumLayers(layers),
        ArbitraryHiddenLayerSize(nPerLayer),
        IsTrained(false)
{
    try
    {
        if (layers == 1) { throw "ANN_INIT_ERROR: What were you thinking? Having a neural network with only 1 layer (0 hidden layers) is no fun!"; }
        else if (layers < 1) { throw "ANN_INIT_ERROR: Umm... you can't have 0 or negative layers... duh..."; }
    }
    catch (string e) { cout << e << endl; }
    if (nPerLayer < 2)
    {
        try { throw "ANN_INIT_ERROR: You really should have at least 2 neurons per layer."; }
        catch (string e) { cout << e << endl; }
    }

    // 1st layer needs to be Matrix(n,m) = (------, x.numOfFeatures)
    // in betweens need to be Matrix(n,m) = (------, ------)
    // last layer needs to be Matrix(n,m) = (y.numOfOutputFeatures, ------)
    int n, m;
    srand((unsigned int) time(0));  // Initializes random number generation
    for (int layer = 0; layer < layers; layer++)
    {
        // If 1st layer...
        if (layer == 0) { m = nInputFeatures; }
        else { m = nPerLayer; }

        // If last layer...
        if (layer == (layers-1)) { n = nOutputs; }
        else { n = nPerLayer; }

        // Randomly initialize W
        //     n+1 rows because last row adds 1 to bottom of output (so next layer can apply weights)
        //     m+1 cols because of weights (W*X = w*x + b*1)
        w[layer] = MatrixXd::Random(n+1,m+1);

        // Set bottom layer to [0,0,...,0,1]
        w[layer].bottomLeftCorner(1,m).setZero();
        w[layer](n,m) = 1;

        cout << "layer " << layer << endl << w[layer] << endl;
    }

}

void NN2::Train(MatrixXd X, MatrixXd Y)
{
    // Each column of X/Y is a (vector of features for) a single sample input/output
    // Each row is the i'th feature

    // Check if dimensions are acceptable
    try
    {
        // Is input.NumSamples == output.NumSamples  ?
        // ... or, does X.NumFeats = NN.NumFeats  ?
        // ... or, does Y.NumFeats = NN.NumOutputs  ??
        if (X.cols() != Y.cols()) {throw "Number of samples (cols) must match between input and output training sets.";}
        else if (X.rows() != this->NumInputFeatures) {throw "Number of input features != specified number of features for this Neural Network.";}
        else if (Y.rows() != this->NumOutputFeatures) {throw "Number of output features != specified number of features for this Neural Network.";}
    }
    catch (string NNTrainingSetException)
    {
        cout << NNTrainingSetException << endl;
    }

    // Now we can begin training!  WOOHOO!!!
    int nInFeats = X.rows();
    int nOutFeats = Y.rows();
    int nSamples = X.cols();
    int nLyrs = this->NumLayers;
    int a = this->ArbitraryHiddenLayerSize;

    float alpha = .01;  // for LeakyReLU, if used
    int iterations = 100000;     // If specified # of iterations for optimization

    MatrixXd XwithBias = BiasAdd(X);
    MatrixXd YwithBias = BiasAdd(Y);
    vector<MatrixXd> lyrCalculations(nLyrs+1);
    lyrCalculations[0] = XwithBias;

    vector<MatrixXd> lyrGradients(nLyrs);
    for (int j = 0; j < iterations; j++)
    {
        // Compute forward
        for (int lyr = 0; lyr < nLyrs; lyr++)
        {
            //L_(j+1) = Sig( w_j * L_j )
            lyrCalculations[lyr+1] = Sigmoid(w[lyr] * lyrCalculations[lyr]); // Get Activations element-by-element
        }

        // Backpropogate
        for (int lyr = nLyrs-1; lyr >= 0; lyr--)
        {
            // Last gradient (Lg_n) must depend on Y
            if (lyr == nLyrs-1)
            {
                lyrGradients[lyr] = MiscOp::MultElByEl(
                        (YwithBias - lyrCalculations[lyr+1]),
                        SigmoidBackGradient(lyrCalculations[lyr+1]));
            }
            else
            {
                lyrGradients[lyr] = MiscOp::MultElByEl(
                        (w[lyr+1].transpose()) * lyrGradients[lyr+1],
                        SigmoidBackGradient(lyrCalculations[lyr+1]));
            }
            w[lyr] += lyrGradients[lyr]*(lyrCalculations[lyr].transpose());
        }

        if ((j+1)%100 == 0)
        {
            cout << "Just finished iteration " << j+1 << endl;
        }
    }

    this->IsTrained = true;
}

MatrixXd NN2::Predict(MatrixXd X)
{
    // Check if trained
    try { if (!IsTrained) {throw "You silly wombat, you! You have attempted to use an untrained NN!";} }
    catch (string e) { cout << e << endl; }

    MatrixXd Xbiased = BiasAdd(X);
    vector<MatrixXd> lyrCalculations(NumLayers+1);
    lyrCalculations[0] = Xbiased;
    for (int lyr = 0; lyr < NumLayers; lyr++)
    {
        lyrCalculations[lyr+1] = Sigmoid(w[lyr] * lyrCalculations[lyr]);
    }

    MatrixXd Y = lyrCalculations[NumLayers];
    MatrixXd Ycorrected = Y.topLeftCorner(Y.rows()-1,Y.cols());
    return Ycorrected;
}
MatrixXd NN2::Sigmoid(MatrixXd input)
{
    long rows = input.rows();
    long cols = input.cols();
    MatrixXd output( rows, cols );

    // Apply to every row EXCEPT bottom - bottom should remain BIAS ones (1 1 ...)
    for (long i = 0; i < rows-1; i++)
    {
        for (long j = 0; j < cols; j++)
        {
            output(i,j) = 1 / (1 + exp(-input(i,j)));
        }
    }
    // Set bottom row to vector of 1's
    output.row(rows-1).setConstant(1);

    return output;
}

MatrixXd NN2::SigmoidBackGradient(MatrixXd input)
{
    long rows = input.rows();
    long cols = input.cols();
    MatrixXd output( rows, cols );
    for (long i = 0; i < rows; i++)
    {
        for (long j = 0; j < cols; j++)
        {
            output(i,j) = input(i,j) * (1 - input(i,j));
        }
    }
    return output;
}

MatrixXd NN2::BiasAdd(MatrixXd input)
{
    MatrixXd output(input.rows()+1, input.cols());
    output << input,
              Eigen::RowVectorXd::Constant(input.cols(),1);
    return output;
}

NN2 NN2::EZTrain(MatrixXd X, MatrixXd Y, int layers, int nPerLayer)
{
    int nInputFeats = X.rows();
    int nOutputFeats = Y.rows();
    NN2 nn(layers, nPerLayer, nInputFeats, nOutputFeats);
    return nn;
}



//MatrixXd NN2::BiasRemove




// X = 0 0 1, 0 1 1, 1 0 1, 1 1 1 .T; // nInFeats = 3
// Y =     0,     1,     1,     0 .T; // nOutputs = 1
// W0 = randomNumsBetween-1&1(---,nInFeats);
// W1 = randomNumsBetween-1&1(nOutFeats,---);
// iterations = 6000;
// for (int j = 0; j < iterations; j++)
//     L1 = activationFunc(W0*X);           // max(alpha*(W*X)),(W*X)) , element-by-element
//     L2 = activationFunc(W1*L1);
//     L2_delta = (Y - L2)*activFuncGradient(L2);   // if W*X>0 : aFV=1, if W*X<=0, aFV=-alpha    , element-by-element
//     L1_delta = (W1.T).dot(L2_delta) * aFG(L1);
//     W1 += L2_delta.dot(L1.T)
//     W0 += L1_delta.dot(X.T)