//
// Created by brahmpowell on 7/20/16.
//

#include "KMeans1.h"
#include <math.h>

using Eigen::ArrayXd;
using namespace std;

KMeans1::KMeans1()
{

}

void KMeans1::Train(ArrayXXd X, int k)
{

    int numSamples = X.cols();          // Each column of X is a data pt (one feature vector)
    int numFeats = X.rows();            // Each row represents a feature (e.g., data with x/y coords would have 2 rows)

    // get range of values
    ArrayXd maxVals = X.rowwise().maxCoeff();  // Max i'th feature value
    ArrayXd minVals = X.rowwise().minCoeff();  // Min i'th feature value
    ArrayXd range = maxVals - minVals;

    cout << "numSamples = " << numSamples << endl;
    cout << "numFeats = " << numFeats << endl;
    cout << "Min vals: " << endl;
    cout << minVals << endl;
    cout << "Max vals: " << endl;
    cout << maxVals << endl;
    cout << "Range: " << endl;
    cout << range << endl;

    srand((unsigned int) time(0));      // Initializes random number generation
    ArrayXXd RandVals = (ArrayXXd::Random(numFeats,k) + ArrayXXd::Constant(numFeats,k,1))/2;
    cout << endl << "Rand vals: " << endl;
    cout << RandVals << endl;

    // Initialize list of centroids
    vector<int> clusterLabels(numSamples);
    vector<int> prevClusterLabels(numSamples);
    c = RandVals;
    for (int j = 0; j < k; j++) { c.col(j) = RandVals.col(j)*range + minVals; } // Make sure centroids are within range

    cout << "centroids: " << endl;
    cout << c << endl;

    bool stillAdjusting = true;
    int iteration = 0;
    while (stillAdjusting)
    {
        iteration++;
        cout << "Iteration " << iteration << " -------------" << endl;
        prevClusterLabels = clusterLabels;
        // Iterate through each sample...
        for (int m = 0; m < numSamples; m++)
        {
            //cout << "sample " << m << ":   ";
            ArrayXd delValues(k);
            // Compare to each centroid...
            for (int i = 0; i < k; i++)
            {
                // Find squared distance
                ArrayXd delValsSingle = X.col(m) - c.col(i);  // Delta vals
                double delValsSquared = (delValsSingle*delValsSingle).sum();
                delValues(i) = sqrt(delValsSquared);//.sum());   // Norm (sqrt(sum of squares)
                //cout << delValues(i) << " ";
            }
            //cout << endl;
            ArrayXd::Index min;         // Index/cluster# of nearest cluster
            delValues.minCoeff(&min);   // ...calculate...
            clusterLabels[m] = min;
            //cout << "    belongs to cluster " << min << endl;
        }

        // Now update the means of all clusters
        ArrayXd total;
        ArrayXd mean;
        int numPts;
        for (int j = 0; j < k; j++)
        {
            total = ArrayXd::Zero(numFeats);
            numPts = 0;
            // Analyze each pt
            for (int m = 0; m < numSamples; m++)
            {
                // Is the point in the cluster?
                if (clusterLabels[m] == j)
                { total += X.col(m); numPts++; }
            }
            mean = total/numPts;
            c.col(j) = mean;
        }

        // Display new centroids
        cout << "centroids: " << endl;
        cout << c << endl;

        // Check to see if all new centroid assignments are the same as previous iteration
        int error;
        bool hasChanged = false;
        int sample = 0;
        while (!hasChanged && sample < numSamples)
        {
            error = clusterLabels[sample] - prevClusterLabels[sample];
            if (error != 0) { hasChanged = true; }
            sample++;
        }
        if (!hasChanged) { stillAdjusting = false; }

    }

}



