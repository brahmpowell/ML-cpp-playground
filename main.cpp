#include <iostream>
#include <vector>
#include "Eigen/Dense"
//#include "CImg/CImg.h"
#include <ImageMagick-7/Magick++.h>
#include "MiscOp.h"
#include "NN1.h"
#include "NN2.h"
#include "KMeans1.h"
using namespace std;
// sudo apt-get install libx11-dev
//using namespace cimg_library;
//using namespace MagickPlusPlus_Header;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXd;
using std::vector;

MatrixXd TestSet()
{
    float inc = .1;

    MatrixXd X(2,21*21);
    for (int x = 0; x < 21; x++)
    {
        for (int y = 0; y < 21; y++)
        {
            X(0,x*21+y) = x*inc;
            X(1,x*21+y) = y*inc;
        }
    }
    return X;
}
void FudgePlot(MatrixXd Y)
{
    for (int i = 20; i >= 0; i--)
    {
        for (int j = 0; j < 21; j++)
        {
            if (Y(0,i*21+j) >= .5)
            {
                cout << "X ";
            }
            else
            {
                cout << "- ";
            }
        }
        cout << endl;
    }
}
void NN_test()
{
    MatrixXd X(2,4);
    MatrixXd Y(1,4);
    X << 0, 0, 1, 1,
            0, 1, 0, 1;
    Y << 0, 1, 1, 0;

    NN2 myNN(4,4,X.rows(),Y.rows());
    myNN.Train(X,Y);

    MatrixXd Xtest = TestSet();
    MatrixXd Ytest = myNN.Predict(Xtest);
    cout << Xtest << endl << endl;

    FudgePlot(Ytest);
}
void KMeansTest()
{
    srand((unsigned int) time(0));      // Initializes random number generation
    MatrixXd Center1 = MatrixXd::Constant(2,20,1);
    MatrixXd Center2 = Center1*2;
    MatrixXd X = MatrixXd::Random(2,40)*.5;
    X.bottomLeftCorner(2,20) += Center1;
    X.bottomRightCorner(2,20) += Center2;
    cout << X << endl;
    //NN_test();
    cout << "---------------------------------------" << endl;
    KMeans1 KK;
    KK.Train(X,2);
}

/*
void CImgTest()
{
    string path = "CImg/examples/img/milla.bmp";
    CImg<unsigned char> image(path.c_str()), visu(500,400,1,3,0);
    //image.load(path);
    const unsigned char red[] = {255, 0, 0}, green[] = {0, 255, 0}, blue[] = {0, 0, 255};
    image.blur(2.5);
    CImgDisplay mainDisp(image, "Click a point"), drawDisp(visu, "Intensity Profile");
    while (!mainDisp.is_closed() && !drawDisp.is_closed())
    {
        mainDisp.wait();
        if (mainDisp.button() && mainDisp.mouse_y()>=0)
        {
            const int y = mainDisp.mouse_y();
            visu.fill(0).draw_graph(image.get_crop(0,y,0,0,image.width()-1,y,0,0),red,1,0,256,0);
            visu.draw_graph(image.get_crop(0,y,0,1,image.width()-1,y,0,1),green,1,0,256,0);
            visu.draw_graph(image.get_crop(0,y,0,2,image.width()-1,y,0,2),blue,1,0,256,0).display(drawDisp);
        }
    }
}*/
/*void TestImageNNTraining()
{
    string path1a = "CImg/examples/img/milla.bmp";
    string path1b = "CImg/examples/img/logo.bmp";
    string path2a = "";
    string path2b = "";

    int numImgs(4); // Number of images is currently 4

    CImg<unsigned char> image1a(path1a.c_str()),
                        image1b(path1b.c_str()),
                        image2a(path2a.c_str()),
                        image2b(path2b.c_str());   //image1.load(path1.c_str());
    image1b.resize(image1a);
    image2a.resize(image1a);
    image2b.resize(image1a);

    int numRows = image1a.height();
    int numCols = image1a.width();
    int dim = numRows*numCols;

    VectorXd img1aVec(dim);
    VectorXd img1bVec(dim);
    VectorXd img2aVec(dim);
    VectorXd img2bVec(dim);

    int vectIndex;
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < numCols; j++)
        {
            vectIndex = numCols*i + j;
            img1a(vectIndex) = image1a(i,j);
            img1b(vectIndex) = image1b(i,j);
            img2a(vectIndex) = image2a(i,j);
            img2b(vectIndex) = image2b(i,j);
        }
    }

    MatrixXd X(dim*2,2);
    MatrixXd Y(1,2);
    Y << 1, 0;
    int layers(5);
    int neuronsPerLayer(dim);
    //NN2.EZTrain(X, Y, layers, neuronsPerLayer)
}*/
int main()
{
    //CImgTest();
    //NN_test();

    Magick::Image myImage();

    return 0;
}

