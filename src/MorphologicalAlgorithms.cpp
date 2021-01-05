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


#include <fstream>;


using namespace cv;
using namespace std;


long EulerNumber(unsigned char ** pix_plane, int mode, int height, int width) {
    unsigned long x, y;
    size_t i;
    // quad-pixel match patterns
    unsigned char Px[] = {
        // P1 - single pixel
        (1 << 3) | (0 << 2) |
        (0 << 1) | (0 << 0),
        (0 << 3) | (1 << 2) |
        (0 << 1) | (0 << 0),
        (0 << 3) | (0 << 2) |
        (1 << 1) | (0 << 0),
        (0 << 3) | (0 << 2) |
        (0 << 1) | (1 << 0),
        // P3 - 3-pixel
        (0 << 3) | (1 << 2) |
        (1 << 1) | (1 << 0),
        (1 << 3) | (0 << 2) |
        (1 << 1) | (1 << 0),
        (1 << 3) | (1 << 2) |
        (0 << 1) | (1 << 0),
        (1 << 3) | (1 << 2) |
        (1 << 1) | (0 << 0),
        // Pd - diagonals
        (1 << 3) | (0 << 2) |
        (0 << 1) | (1 << 0),
        (0 << 3) | (1 << 2) |
        (1 << 1) | (0 << 0)
    };
    unsigned char Imq;
    // Pattern match counters
    long C1 = 0, C3 = 0, Cd = 0;

    assert ( (mode == 4 || mode == 8) && "Calling EulerNumber with mode other than 4 or 8");




    ofstream alaki;
    alaki.open("Morphology.csv");




    // update pattern counters by scanning the image.
    for (y = 1; y < height; y++) {
        for (x = 1; x < width; x++) {
           //?? does not have meaning here    if(std::isnan(pix_plane[y][x])) continue; //MM
            // Get the quad-pixel at this image location
            Imq = 0;
            if (pix_plane[y-1][x-1] > 0) Imq |=  (1 << 3);
            if (pix_plane[y-1][x] > 0) Imq |=  (1 << 2);
            if (pix_plane[y][x-1] > 0) Imq |=  (1 << 1);
            if (pix_plane[y][x] > 0) Imq |=  (1 << 0);
            // find the matching pattern
            for (i = 0; i < 10; i++) if (Imq == Px[i]) break;
            // unsigned i always >= 0
            // if      (i >= 0 && i <= 3) C1++;
            if      (i <= 3) C1++;
            else if (i >= 4 && i <= 7) C3++;
            else if (i == 8 && i == 9) Cd++;

            alaki<<y<<"  "<<x<<"  "<<Imq<<"  "<<C1<<"  "<<C3<<"  "<<Cd<<std::endl;

        }
    }

alaki.close();

/*
   //MM: We need to take into consideration the first column and the first row of the image when nan values exists
   // as they might not be included by the neighboring pixels in the next row/column. The neighboring Pixles might be
   //excluded in the computations due to their nan values
        for (x = 0; x < width-1; x++) {
            if(std::isnan(pix_plane[0][x])) continue; //MM
            // Get the quad-pixel at this image location
            Imq = 0;
            if (pix_plane[0][x] > 0) {
                if (!pix_plane[1][x]> 0 || !pix_plane[1][x+1]>0 ) Imq |=  (1 << 0);
            }
            // find the matching pattern
            for (i = 0; i < 10; i++) if (Imq == Px[i]) break;
            // unsigned i always >= 0
            // if      (i >= 0 && i <= 3) C1++;
            if (i <= 3) C1++;
            else if (i >= 4 && i <= 7) C3++;
            else if (i == 8 && i == 9) Cd++;
        }
        for (y = 0; y < height-1; y++) {
            if(std::isnan(pix_plane[y][0])) continue; //MM
            // Get the quad-pixel at this image location
            Imq = 0;
            if (pix_plane[y][0] > 0) {
                if (!pix_plane[y][1] > 0 || !pix_plane[y+1][1]>0 ) Imq |=  (1 << 0);
            }
            // find the matching pattern
            for (i = 0; i < 10; i++) if (Imq == Px[i]) break;
            // unsigned i always >= 0
            // if      (i >= 0 && i <= 3) C1++;
            if (i <= 3) C1++;
            else if (i >= 4 && i <= 7) C3++;
            else if (i == 8 && i == 9) Cd++;
        }
*/
    if (mode == 4)
        return ( (C1 - C3 + (2*Cd)) / 4);
    else
        return ( (C1 - C3 - (2*Cd)) / 4);
}



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



/* the input should be a binary image */
void WeightedGlobalCentroid(const ImageMatrix &Im, double *x_centroid, double *y_centroid) {
    unsigned int x,y,w = Im.width, h = Im.height;
    double x_mass=0,y_mass=0,mass=0;
    readOnlyPixels pix_plane = Im.ReadablePixels();

    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            if (pix_plane(y,x) > 0) {
                x_mass=x_mass+(x+1)*pix_plane(y,x);    /* the "+1" is only for compatability with matlab code (where index starts from 1) */
                y_mass=y_mass+(y+1)*pix_plane(y,x);    /* the "+1" is only for compatability with matlab code (where index starts from 1) */
                mass+=pix_plane(y,x);
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

WeightedGlobalCentroid(Im,&ratios[50],&ratios[51]);

//--------------
double mean= Im.stats.mean();

double max= Im.stats.max(); //int max??
double min= Im.stats.min(); //int  min??
double median=Im.get_median();









//--------------
//double* arr = new double[Im.height*Im.width];
uchar* arr = new uchar[Im.height*Im.width];

readOnlyPixels in_plane = Im.ReadablePixels();

double sqrdTmp=0;
double TrpdTmp=0;
double QuadTmp=0;

for (unsigned int y = 0; y < Im.height; ++y)
    for (unsigned int x = 0; x < Im.width; ++x){
        double PixelVal= in_plane (y,x);
        if (std::isnan(PixelVal)) arr[y*Im.width+x]=0;
        else
        {
            arr[y*Im.width+x]=1;
            double tmp= PixelVal-mean;
            sqrdTmp += tmp*tmp;
            TrpdTmp += tmp*tmp*tmp;
            QuadTmp += tmp*tmp*tmp*tmp;
        }
    }
//--------------
double Variance= sqrdTmp/Im.stats.n(); //(Im.stats.n()-1)
double STDEV= sqrt(Variance);
ratios[100]=STDEV;
double Skewness= (TrpdTmp/Im.stats.n())/pow(STDEV,3);
ratios[101]=Skewness;
double Kurtosis= (QuadTmp/Im.stats.n())/pow(Variance,2) - 3;
ratios[102]=Kurtosis;
//--------------

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

//cout<<contours.size()<<endl;

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

//The coordinates of convexHull Points
for (int i = 0; i < hull[0].size(); i++) {
    Point coordinate_i_ofcontour = hull[0][i];
    cout << endl << "contour with coordinates: x = " << coordinate_i_ofcontour.x << " y = " << coordinate_i_ofcontour.y;
}






//fitEllipse()
/*vector<RotatedRect> minRect( contours.size() );
vector<RotatedRect> minEllipse( contours.size() );
for( size_t i = 0; i < contours.size(); i++ )
{
    minRect[i] = minAreaRect( contours[i] ); //minAreaRect Finds a rotated rectangle of the minimum area enclosing the input 2D point set.
    if( contours[i].size() > 5 )
    {
        minEllipse[i] = fitEllipse( contours[i] );
    }
}
Point2f minEllipse_rect_points[4];
minEllipse[0].points(minEllipse_rect_points);
for ( int j = 0; j < 4; j++ ){
    ratios[5+j*2]= static_cast<double>(minEllipse_rect_points[j].x);
    ratios[5+j*2+1]= static_cast<double>(minEllipse_rect_points[j].y);
}

Point2f centerPoint= minEllipse[0].center;

ratios[20]=centerPoint.x;
ratios[21]=centerPoint.y;

ratios[22]=minEllipse[0].angle; //Orientation

float a     = minEllipse[0].size.width  / 2;
float b     = minEllipse[0].size.height / 2;
if (a>b) {
    ratios[23]=a; //Major Semi-Axis
    ratios[24]=b; //Minor Semi-Axis
    ratios[25]=sqrt(1-(b*b)/(a*a)); //eccentricity
}
else {
    ratios[23]=b; //Major Semi-Axis
    ratios[24]=a; //Minor Semi-Axis
    ratios[25]=sqrt(1-(a*a)/(b*b)); //eccentricity
}

Mat drawing2 = Mat::zeros( matrix.size(), CV_8UC3 );
for( size_t i = 0; i< contours.size(); i++ )
{
    Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
    // contour
  //  drawContours( drawing2, contours, (int)i, color );
    // ellipse
    ellipse( drawing2, minEllipse[i], color, 2 );
    // rotated rectangle
    Point2f rect_points[4];
    minRect[i].points( rect_points );
    for ( int j = 0; j < 4; j++ )
    {
     //   line( drawing2, rect_points[j], rect_points[(j+1)%4], color );
    }
}
//imshow( "Contours", drawing2 );
//waitKey();

*/



//Reconstruct LabelledImage

unsigned char ** LabeledImageMatrix = new unsigned char *[Im.height];
for (int i=0; i<Im.height; ++i) LabeledImageMatrix[i] = new unsigned char [Im.width];


//??  copyFields (matrix_IN);
//??  allocate (matrix_IN.width, matrix_IN.height);

//writeablePixels Labeled_plane = LabeledImageMatrix.WriteablePixels();
//readOnlyPixels in_plane = matrix_IN.ReadablePixels();

//writeablePixels pix_plane = ROI_Bounding_Box.WriteablePixels();
//readOnlyPixels in_plane = image_matrix.ReadablePixels();

/* classify the pixels by the threshold */
/*for (unsigned int a = 0; a < width*height; a++){
    if (std::isnan(in_plane.array().coeff(a))) {(out_plane.array())(a)=in_plane.array().coeff(a); continue;} //MM

    if (in_plane.array().coeff(a) > OtsuGlobalThreshold) (out_plane.array())(a) = stats.add (1);
    else (out_plane.array())(a) = stats.add (0);
}

*/
        for (int y = 0; y < Im.height; ++y)
            for (int x = 0; x < Im.width; ++x){
//                if (std::isnan(in_plane (y,x))) LabeledImageMatrix[y][x]=std::numeric_limits<double>::quiet_NaN();
                if (std::isnan(in_plane (y,x))) LabeledImageMatrix[y][x]=(unsigned char)0;
                else LabeledImageMatrix[y][x]=(unsigned char)1;
            }

long Euler= EulerNumber(LabeledImageMatrix,8,Im.height,Im.width);








float* Intensity = new float [Im.height*Im.width];

for (unsigned int y = 0; y < Im.height; ++y)
    for (unsigned int x = 0; x < Im.width; ++x){
       // if (std::isnan(in_plane (y,x)))
            Intensity[y*Im.width+x]=in_plane (y,x);
    }

cv::Mat IntensityMatrix = cv::Mat(Im.height,Im.width,CV_32FC1,Intensity);


Mat hist;

//int channels[] = {0};
    int histSize[] = {65090};
    float range[] = { 0, 65090 }; //
    const float* ranges[] = { range };

    calcHist( &IntensityMatrix, 1, 0, matrix, // do not use mask
         hist, 1, histSize, ranges,
         true, // the histogram is uniform
         false );

   Mat histNorm = hist / (float)Im.stats.n();

   int cntt=0;
for (int i=0; i<hist.rows; i++) if (hist.at<float>(i,0)!=0) {
// auto aa= hist.at<float>(i,0);
 //cntt+=aa;
 //auto aaa= hist.at<float>(i,0);
}

float MaxValue=0;
int maxBinIndex=-1;
   double entropy = 0.0;
   for (int i=0; i<histNorm.rows; i++)
   {
       float binEntry = histNorm.at<float>(i,0);
       if (binEntry != 0.0){
           entropy -= binEntry * log2(binEntry);
       if (binEntry>MaxValue) {MaxValue=binEntry; maxBinIndex=i;}
       }
   }


ratios[60]=maxBinIndex; //mode value











cout << "Finished!\n" << endl;


delete [] arr;

}










