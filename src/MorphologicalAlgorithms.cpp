/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* Written by:                                                                   */
/*        Mahdi Maghrebi <mahdi.maghrebi [at] nih [dot] gov>             */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include "MorphologicalAlgorithms.h"
#include "cmatrix.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

/* the input should be a binary image */
void GlobalCentroid2(const ImageMatrix &Im, double *x_centroid, double *y_centroid) {
    unsigned int x,y,w = Im.width, h = Im.height;
    double x_mass=0,y_mass=0,mass=0;
    readOnlyPixels pix_plane = Im.ReadablePixels();

    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            if (pix_plane(y,x) > 0) {
                x_mass=x_mass+x+1;    /* the "+1" is only for compatability with matlab code (where index starts from 1) */
                y_mass=y_mass+y+1;    /* the "+1" is only for compatability with matlab code (where index starts from 1) */
                mass++;
            }
    if (mass) {
        *x_centroid=x_mass/mass;
        *y_centroid=y_mass/mass;
    } else *x_centroid=*y_centroid=0;
}


void MorphologicalAlgorithms(const ImageMatrix &Im, double *ratios){

//----------Total Number of Pixels in ROI
ratios[0]=Im.stats.n();

//---------------Position and size of the smallest box containing the region--------------
ratios[1]=Im.ROIWidthBeg;
ratios[2]=Im.ROIHeightBeg;
ratios[3]=Im.width;
ratios[4]=Im.height;

//--------------centroids
GlobalCentroid2(Im,&ratios[5],&ratios[6]);



//double* arr = new double[Im.height*Im.width];
uchar* arr = new uchar[Im.height*Im.width];

readOnlyPixels in_plane = Im.ReadablePixels();

for (unsigned int y = 0; y < Im.height; ++y)
    for (unsigned int x = 0; x < Im.width; ++x){
        if (std::isnan(in_plane (y,x)))
            arr[y*Im.width+x]=0;
        else arr[y*Im.width+x]=1;
    }


//cv::Mat matrix = cv::Mat(Im.height,Im.width,CV_64FC1,(uchar*)arr);
cv::Mat matrix = cv::Mat(Im.height,Im.width,CV_8UC1,arr);

if( matrix.empty() )
{
    cout << "Could not find the image!\n" << endl;
    return ;
}

vector<vector<Point>> contours;

findContours( matrix, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE );

RNG rng(12345);

vector<vector<Point>> hull(contours.size() );

cout<<contours.size()<<endl;

for( size_t i = 0; i < contours.size(); i++ )
{
    convexHull( contours[i], hull[i] );
}

Mat drawing = Mat::zeros( matrix.size(), CV_8UC3 );
for( size_t i = 0; i< contours.size(); i++ )
{
    Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
//    drawContours( drawing, contours, (int)i, color );
    drawContours( drawing, hull, (int)i, color );
}
// imshow( "Hull demo", drawing );
// waitKey();




double ROIArea = contourArea(contours[0]);
Rect boundingBox = boundingRect(matrix);
int boundingBoxArea=boundingBox.width*boundingBox.height;
double extent = ROIArea/boundingBoxArea;

double convexHullArea = contourArea(hull[0]);
double solidity = ROIArea/convexHullArea;

//cout<<ROIArea<<"  "<<boundingBoxArea<<"  "<<extent<<"  "<<convexHullArea<<"  "<<solidity<<endl;

double AspectRatio=(float)boundingBox.width/(float)boundingBox.height;

double EquivalentDiameter= sqrt(4/M_PI*ROIArea);
//cout<< M_PI<<"  "<<AspectRatio<<"  "<<EquivalentDiameter<<endl;

double ROIPerimeter = arcLength(contours[0],true);
double Circularity=4*M_PI*ROIArea/(ROIPerimeter*ROIPerimeter);
cout<< ROIPerimeter<<"  "<<Circularity<<endl;


for (int i = 0; i < hull[0].size(); i++) {
    Point coordinate_i_ofcontour = hull[0][i];
    cout << endl << "contour with coordinates: x = " << coordinate_i_ofcontour.x << " y = " << coordinate_i_ofcontour.y;
}


//fitEllipse()


cout << "Finished!\n" << endl;


delete [] arr;

}










