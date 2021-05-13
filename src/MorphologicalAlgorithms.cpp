/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* Written by:                                                                   */
/*        Mahdi Maghrebi <mahdi.maghrebi [at] nih [dot] gov>                     */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include "MorphologicalAlgorithms.h"
#include "cmatrix.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
//#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <vector>
#include <math.h>
#include <memory>
#include <tiffio.h>

using namespace std;
using namespace cv;

long EulerNumber(unsigned char * arr, int mode, int height, int width) {
    unsigned long x, y;
    size_t i;
    // quad-pixel match patterns
    unsigned char Px[] = { //MM: 0 or 1 in the left side of << represent binary pixel values
                           // P1 - single pixel  8/4/2/1
                           (1 << 3) | (0 << 2) |
                           (0 << 1) | (0 << 0),
                           (0 << 3) | (1 << 2) |
                           (0 << 1) | (0 << 0),
                           (0 << 3) | (0 << 2) |
                           (1 << 1) | (0 << 0),
                           (0 << 3) | (0 << 2) |
                           (0 << 1) | (1 << 0),
                           // P3 - 3-pixel   7/11/13/14
                           (0 << 3) | (1 << 2) |
                           (1 << 1) | (1 << 0),
                           (1 << 3) | (0 << 2) |
                           (1 << 1) | (1 << 0),
                           (1 << 3) | (1 << 2) |
                           (0 << 1) | (1 << 0),
                           (1 << 3) | (1 << 2) |
                           (1 << 1) | (0 << 0),
                           // Pd - diagonals  9/6
                           (1 << 3) | (0 << 2) |
                           (0 << 1) | (1 << 0),
                           (0 << 3) | (1 << 2) |
                           (1 << 1) | (0 << 0)
                         };
    unsigned char Imq;
    // Pattern match counters
    long C1 = 0, C3 = 0, Cd = 0;

    assert ( (mode == 4 || mode == 8) && "Calling EulerNumber with mode other than 4 or 8");

    // update pattern counters by scanning the image.
    for (y = 1; y < height; y++) {
        for (x = 1; x < width; x++) {
            // Get the quad-pixel at this image location
            Imq = 0;
            if (arr[(y-1)*width+x-1] > 0) Imq |=  (1 << 3);
            if (arr[(y-1)*width+x] > 0) Imq |=  (1 << 2);
            if (arr[y*width+x-1] > 0) Imq |=  (1 << 1);
            if (arr[y*width+x] > 0) Imq |=  (1 << 0);

            // find the matching pattern
            for (i = 0; i < 10; i++) if (Imq == Px[i]) break;
            // unsigned i always >= 0
            // if      (i >= 0 && i <= 3) C1++;
            if      (i <= 3) C1++;
            else if (i >= 4 && i <= 7) {
                C3++;}
            else if (i == 8 && i == 9) {  //  ||??
                Cd++;}
        }
    }

    if (mode == 4)
        return ( (C1 - C3 + (2*Cd)) / 4);
    else
        return ( (C1 - C3 - (2*Cd)) / 4);
}

/* the input should be a binary image */
void GlobalCentroid2(const ImageMatrix &Im, double * x_centroid, double * y_centroid) {
    unsigned int x,y,w = Im.width, h = Im.height;
    double x_mass=0,y_mass=0,mass=0;
    readOnlyPixels pix_plane = Im.ReadablePixels();

    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            if (!std::isnan(pix_plane(y,x))){
                x_mass=x_mass+x+1;    /* the "+1" is only for compatability with matlab code (where index starts from 1) */
                y_mass=y_mass+y+1;    /* the "+1" is only for compatability with matlab code (where index starts from 1) */
                mass++;
            }
    if (mass) {
        *x_centroid=x_mass/mass+Im.ROIWidthBeg;
        *y_centroid=y_mass/mass+Im.ROIHeightBeg;
    } else *x_centroid=*y_centroid=0;
}

void Extrema (const ImageMatrix& Im, double *ratios){

    int TopMostIndex=Im.height;
    int LowestIndex= 0;
    int LeftMostIndex=Im.width;
    int RightMostIndex=0;

    readOnlyPixels in_plane = Im.ReadablePixels();

    for (unsigned int y = 0; y < Im.height; ++y){
        for (unsigned int x = 0; x < Im.width; ++x){
            if (!std::isnan(in_plane(y,x))){
                if (y+1<TopMostIndex) TopMostIndex=y+1;       //MATLAB indices start from 1
                if (y+1>LowestIndex) LowestIndex=y+1;         //MATLAB indices start from 1
                if (x+1<LeftMostIndex) LeftMostIndex=x+1;     //MATLAB indices start from 1
                if (x+1>RightMostIndex) RightMostIndex=x+1;   //MATLAB indices start from 1
            }
        }
    }

    int TopMost_MostLeftIndex=-1;
    int TopMost_MostRightIndex=-1;

    for (unsigned int x = 0; x < Im.width; ++x){
        if (!std::isnan(in_plane(TopMostIndex-1,x))){ TopMost_MostLeftIndex=x+1; break;}
    }

    for (unsigned int x = Im.width-1; x >= 0; --x){
        if (!std::isnan(in_plane(TopMostIndex-1,x))){ TopMost_MostRightIndex=x+1; break;}
    }

    int Lowest_MostLeftIndex=-1;
    int Lowest_MostRightIndex=-1;

    for (unsigned int x = 0; x < Im.width; ++x){
        if (!std::isnan(in_plane(LowestIndex-1,x))){ Lowest_MostLeftIndex=x+1; break;}
    }

    for (unsigned int x = Im.width-1; x >= 0; --x){
        if (!std::isnan(in_plane(LowestIndex-1,x))){ Lowest_MostRightIndex=x+1; break;}
    }

    int LeftMost_Top=-1;
    int LeftMost_Bottom=-1;

    for (unsigned int y = 0; y < Im.height; ++y){
        if (!std::isnan(in_plane(y,LeftMostIndex-1))){ LeftMost_Top=y+1; break;}
    }

    for (unsigned int y = Im.height-1; y >= 0; --y){
        if (!std::isnan(in_plane(y,LeftMostIndex-1))){ LeftMost_Bottom=y+1; break;}
    }

    int RightMost_Top=-1;
    int RightMost_Bottom=-1;

    for (unsigned int y = 0; y < Im.height; ++y){
        if (!std::isnan(in_plane(y,RightMostIndex-1))){ RightMost_Top=y+1; break;}
    }

    for (unsigned int y = Im.height-1; y >= 0; --y){
        if (!std::isnan(in_plane(y,RightMostIndex-1))){ RightMost_Bottom=y+1; break;}
    }

    float P1y = TopMostIndex-0.5+Im.ROIHeightBeg;
    float P1x = TopMost_MostLeftIndex-0.5+Im.ROIWidthBeg;

    float P2y = TopMostIndex-0.5+Im.ROIHeightBeg;
    float P2x = TopMost_MostRightIndex+0.5+Im.ROIWidthBeg;

    float P3y = RightMost_Top-0.5+Im.ROIHeightBeg;//
    float P3x = RightMostIndex+0.5+Im.ROIWidthBeg;

    float P4y = RightMost_Bottom+0.5+Im.ROIHeightBeg;
    float P4x = RightMostIndex+0.5+Im.ROIWidthBeg;

    float P5y = LowestIndex+0.5+Im.ROIHeightBeg;
    float P5x = Lowest_MostRightIndex+0.5+Im.ROIWidthBeg;

    float P6y = LowestIndex+0.5+Im.ROIHeightBeg;
    float P6x = Lowest_MostLeftIndex-0.5+Im.ROIWidthBeg;

    float P7y = LeftMost_Bottom+0.5+Im.ROIHeightBeg;
    float P7x = LeftMostIndex-0.5+Im.ROIWidthBeg;

    float P8y = LeftMost_Top-0.5+Im.ROIHeightBeg;
    float P8x = LeftMostIndex-0.5+Im.ROIWidthBeg;

    ratios[8]=P1x;
    ratios[9]=P2x;
    ratios[10]=P3x;
    ratios[11]=P4x;
    ratios[12]=P5x;
    ratios[13]=P6x;
    ratios[14]=P7x;
    ratios[15]=P8x;
    ratios[16]=P1y;
    ratios[17]=P2y;
    ratios[18]=P3y;
    ratios[19]=P4y;
    ratios[20]=P5y;
    ratios[21]=P6y;
    ratios[22]=P7y;
    ratios[23]=P8y;

    return;
}

Statistics ComputeCommonStatistics (vector<int> Data){

    Statistics output;

    output.max = *max_element(Data.begin(), Data.end());
    output.min = *min_element(Data.begin(), Data.end());

    double sum = 0;
    for (int i=0; i< Data.size(); i++) sum += Data[i];
    output.mean = sum / Data.size();

    double sumSqrd=0;
    for (int i=0; i< Data.size(); i++) sumSqrd += (Data[i]-output.mean)*(Data[i]-output.mean);
    output.stdev = sqrt(sumSqrd/Data.size());

    //Make a Histogram
    int intMax=(int)ceil(output.max);
    int intMin=(int)floor(output.min);
    int binCounts=intMax-intMin+1;
    int* histBins =new int [binCounts];

    for (int i=0; i<binCounts; ++i) histBins[i]=0;
    for (int i=0; i<Data.size(); i++) ++ histBins[Data[i]-intMin];

    double MaxValue=0;
    int maxBinIndex=-1;
    //Loop over all the bins
    for (int i=0; i<binCounts; i++) {
        if (histBins[i] > MaxValue) {MaxValue=histBins[i]; maxBinIndex=i;}
    }
    output.mode = maxBinIndex+intMin;


    std::sort(Data.begin(),Data.end());
    double median;
    size_t half = Data.size() / 2;
    if (Data.size() % 2 == 0) {
        nth_element(Data.begin(), Data.begin()+half, Data.end());
        median = Data[half];
        nth_element(Data.begin(), Data.begin()+half-1, Data.end());
        median += (Data[half-1]);
        median /= 2.0;
    } else {
        nth_element(Data.begin(), Data.begin()+half, Data.end());
        median = Data[half];
    }
    output.median=median;

    return output;
}

Statistics ComputeCommonStatistics2 (vector<double> Data){

    Statistics output;

    output.max = *max_element(Data.begin(), Data.end());
    output.min = *min_element(Data.begin(), Data.end());

    double sum = 0;
    for (int i=0; i< Data.size(); i++) sum += Data[i];
    output.mean = sum / Data.size();

    double sumSqrd=0;
    for (int i=0; i< Data.size(); i++) sumSqrd += (Data[i]-output.mean)*(Data[i]-output.mean);
    output.stdev = sqrt(sumSqrd/Data.size());

    //Make a Histogram
    int intMax=(int)ceil(output.max);
    int intMin=(int)floor(output.min);
    int binCounts=intMax-intMin+1;
    int* histBins =new int [binCounts];

    for (int i=0; i<binCounts; ++i) histBins[i]=0;
    for (int i=0; i<Data.size(); i++) ++ histBins[(int)Data[i]-intMin];

    double MaxValue=0;
    int maxBinIndex=-1;
    //Loop over all the bins
    for (int i=0; i<binCounts; i++) {
        if (histBins[i] > MaxValue) {MaxValue=histBins[i]; maxBinIndex=i;}
    }
    output.mode = maxBinIndex+intMin;

    std::sort(Data.begin(),Data.end());
    double median;

    size_t half = Data.size() / 2;
    if (Data.size() % 2 == 0) {
        nth_element(Data.begin(), Data.begin()+half, Data.end());
        median = Data[half];
        nth_element(Data.begin(), Data.begin()+half-1, Data.end());
        median += (Data[half-1]);
        median /= 2.0;
    } else {
        nth_element(Data.begin(), Data.begin()+half, Data.end());
        median = Data[half];
    }

    output.median=median;

    return output;
}

void MorphologicalAlgorithms(const ImageMatrix &Im, double *ratios){
    double UnitConversion=Im.PixelsUnit;

    //----------Total Number of ROI Pixels--------------
    int PixelsCount=Im.stats.n();
    ratios[0]=(double)PixelsCount;

    //---------------Position and size of the smallest box containing the region (Bounding Box)---------
    //For consistency with MATLAB 1 and 0.5 was added below. 1 accounts for pixel index which starts from 1 in MATLAB
    //and 0.5 accounts for the pixel side rather than its center
    double BoundingBoxWidthBeg=Im.ROIWidthBegActual+1-0.5;
    double BoundingBoxHeightBeg=Im.ROIHeightBegActual+1-0.5;
    double boundingBoxArea=Im.ROIWidthActual*Im.ROIHeightActual;

    ratios[1]=BoundingBoxWidthBeg;
    ratios[2]=BoundingBoxHeightBeg;
    ratios[3]=Im.ROIWidthActual;
    ratios[4]=Im.ROIHeightActual;
    ratios[5]=boundingBoxArea;

    //--------------Spatial Centroids----------------
    double xCentroid,yCentroid;
    GlobalCentroid2(Im,&xCentroid,&yCentroid);
    ratios[6]=xCentroid;
    ratios[7]=yCentroid;

    //--------------Make a binary array-----------
    uchar * arr = new uchar[Im.height*Im.width];

    readOnlyPixels in_plane = Im.ReadablePixels();

    for (unsigned int y = 0; y < Im.height; ++y)
        for (unsigned int x = 0; x < Im.width; ++x){
            if (std::isnan(in_plane (y,x))) arr[y*Im.width+x]=0;
            else arr[y*Im.width+x]=1;
        }
    //--------------Fitting an Ellipse--------------------------------
    //Reference: https://www.mathworks.com/matlabcentral/mlc-downloads/downloads/submissions/19028/versions/1/previews/regiondata.m/index.html

    // Calculate normalized second central moments for the region.
    // 1/12 is the normalized second central moment of a pixel with unit length.
    double XSquaredTmp=0,YSquaredTmp=0, XYSquaredTmp=0;
    for (unsigned int y = 0; y < Im.height; ++y)
        for (unsigned int x = 0; x < Im.width; ++x){
            if (!std::isnan(in_plane (y,x))) {
                XSquaredTmp += (double(x)-(xCentroid-Im.ROIWidthBeg))*(double(x)-(xCentroid-Im.ROIWidthBeg));
                //Y in below is negative for the orientation calculation (measured in the counter-clockwise direction).
                YSquaredTmp += (-double(y)+(yCentroid-Im.ROIHeightBeg))*(-double(y)+(yCentroid-Im.ROIHeightBeg));
                XYSquaredTmp += (double(x)-(xCentroid-Im.ROIWidthBeg))*(-double(y)+(yCentroid-Im.ROIHeightBeg));
            }
        }
    double uxx= XSquaredTmp/Im.stats.n()+1.0/12.0;
    double uyy= YSquaredTmp/Im.stats.n()+1.0/12.0;
    double uxy= XYSquaredTmp/Im.stats.n();

    // Calculate major axis length, minor axis length, and eccentricity.
    double common = sqrt((uxx - uyy)*(uxx - uyy) + 4*uxy*uxy);
    double MajorAxisLength = 2*sqrt(2)*sqrt(uxx + uyy + common);
    double MinorAxisLength = 2*sqrt(2)*sqrt(uxx + uyy - common);
    double Eccentricity = 2*sqrt((MajorAxisLength/2)*(MajorAxisLength/2)-(MinorAxisLength/2)*(MinorAxisLength/2))/MajorAxisLength;

    // Calculate orientation [-90,90]
    double num,den,Orientation;
    if (uyy > uxx) {
        num = uyy - uxx + sqrt((uyy - uxx)*(uyy - uxx) + 4*uxy*uxy);
        den = 2*uxy;
    }
    else {
        num = 2*uxy;
        den = uxx - uyy + sqrt((uxx - uyy)*(uxx - uyy) + 4*uxy*uxy);
    }
    if (num == 0 && den == 0) Orientation = 0;
    else Orientation = (180/M_PI) * atan(num/den);

    //--------------Coordinates of the Extrema Pixels--------
    Extrema (Im, ratios);

    //--------------Computing dimensions features--------------------------------
    cv::Mat matrix = cv::Mat(Im.height,Im.width,CV_8UC1,arr);

    if( matrix.empty() ) {
        cout << "Could not find the image!\n" << endl;
        return ;
    }


    //Store all the Features in one vector, one per theta (rotation angel)
    vector<int> ChordAll;
    vector<int> LongestChordAll;
    vector<int> MartinLengthAll;
    vector<int> NassensteinDiameterAll;
    vector<int> thetas;

    for (int theta=0; theta<180;theta++){
        thetas.push_back(theta);

        //First we need to rotate the image
        //https://stackoverflow.com/questions/22041699/rotate-an-image-without-cropping-in-opencv-in-c

        //To avoid cropping of the rotated image, a large enough dimension should be considered for it.
        //diagonal of the non-rotated image is large enough for this purpose
        int diagonal = (int)sqrt(matrix.cols*matrix.cols+matrix.rows*matrix.rows);
        int newWidth = diagonal;
        int newHeight =diagonal;

        int offsetX = (newWidth - matrix.cols) / 2;
        int offsetY = (newHeight - matrix.rows) / 2;
        Mat targetMat(newWidth, newHeight, matrix.type());
        Point2f matrix_center(targetMat.cols/2.0F, targetMat.rows/2.0F);

        matrix.copyTo(targetMat.rowRange(offsetY, offsetY + matrix.rows).colRange(offsetX, offsetX + matrix.cols));
        Mat rot_mat = getRotationMatrix2D(matrix_center, theta, 1.0);

        Mat RotatedImage;
        warpAffine(targetMat, RotatedImage, rot_mat, targetMat.size());

        vector<Point> nz;
        findNonZero(RotatedImage,nz); //non-zero elements of the rotated image

        vector<int> * yCoordinatePoints ; //y coordinates of the start/end points of chords in each row of the rotated image
        yCoordinatePoints = new vector<int> [RotatedImage.rows];

        for (int i = 0; i < nz.size(); i++){  //nz[i].y = row index, nz[i].x= column index
            yCoordinatePoints[nz[i].y].push_back(nz[i].x);
        }

        //----------------------------------------------
        //Now, lets compute Longest Chord
        int LongestChord =-1;

        int Chord;
        for (int i = 0; i < RotatedImage.rows; i++){
            if (yCoordinatePoints[i].size() == 0) continue;
            //If no discontinuty exists
            int min_y = *min_element(yCoordinatePoints[i].begin(), yCoordinatePoints[i].end());
            int max_y = *max_element(yCoordinatePoints[i].begin(), yCoordinatePoints[i].end());
            if ((max_y-min_y+1) == yCoordinatePoints[i].size()) {Chord= max_y-min_y+1; if (LongestChord < Chord) LongestChord = Chord; ChordAll.push_back(Chord);}
            //If discontinuity exists in each row of the image
            else {
                std::sort(yCoordinatePoints[i].begin(),yCoordinatePoints[i].end());//Sorting the vector
                Chord=1;
                for (int j = 0; j < yCoordinatePoints[i].size()-1; j++){
                    //If continuity persists
                    if (yCoordinatePoints[i][j] + 1 == yCoordinatePoints[i][j+1]) Chord++;
                    else {
                        //reset the chord length as soon as the first discontinuity met in the row
                        if (LongestChord < Chord) LongestChord = Chord;
                        ChordAll.push_back(Chord);
                        Chord=1;
                    }
                }
                ChordAll.push_back(Chord);
                if (LongestChord < Chord) LongestChord = Chord;
            }
        }
        LongestChordAll.push_back(LongestChord);

        //----------------------------------------------
        //Now, lets compute Martin Length
        int MartinLength;

        float MartinSplitRatio=0.5;
        int RequiredObjectArea= (int) ceil(MartinSplitRatio * nz.size());
        int MetObjectArea=0;

        for (int i = 0; i < RotatedImage.rows; i++){
            MetObjectArea += yCoordinatePoints[i].size();

            if (MetObjectArea > RequiredObjectArea){
                //Martin Diameter is on this row of the image

                int min_y = *min_element(yCoordinatePoints[i].begin(), yCoordinatePoints[i].end());
                int max_y = *max_element(yCoordinatePoints[i].begin(), yCoordinatePoints[i].end());
                MartinLength= max_y-min_y+1;
                break;
            }
        }
        MartinLengthAll.push_back(MartinLength);

        //----------------------------------------------
        //Now, lets compute Nassenstein Diameter
        int lowestRowIndex=-1;
        for (int i = RotatedImage.rows-1; i >= 0 ; i--){
            if (yCoordinatePoints[i].size() != 0) {
                lowestRowIndex = i;
                break;
            }
        }

        if (lowestRowIndex == -1) cout<<" Error in Nassenstein Diameter algorithm: No lowestRowIndex found"<<endl;

        std::sort(yCoordinatePoints[lowestRowIndex].begin(),yCoordinatePoints[lowestRowIndex].end()); //Sorting the vector

        if (yCoordinatePoints[lowestRowIndex].size() == 0) cout<<"Error in Nassenstein Diameter algorithm: No element found in the lowest row"<<endl;

        //Accroding to imea library: There might be several touching points in the lowest row and
        // thus the center of first continuous contact surface from left is selected here

        //BeginningYIndex is the first element touching the bottom in the lowest row of the rotated image
        int BeginningYIndex = yCoordinatePoints[lowestRowIndex][0];

        int ContinuingPixelsCounts=0;
        for (int j = 0; j < yCoordinatePoints[lowestRowIndex].size()-1; j++){
            if (yCoordinatePoints[lowestRowIndex][j] + 1 == yCoordinatePoints[lowestRowIndex][j+1]) ContinuingPixelsCounts++;
        }
        //Now find TargetColumn where Nassenstein Diameter is coming from it
        int TargetColumn= (int) ceil(BeginningYIndex + ((float)ContinuingPixelsCounts/2));

        //Now, identify the non-zero elements in TargetColumn
        cv::Mat TargetColumnMat = RotatedImage.col(TargetColumn);

        vector<Point> nz_TargetColumnMat;
        findNonZero(TargetColumnMat,nz_TargetColumnMat); //non-zero elements in TargetColumn of the rotated image

        vector<int> xCoordinatePoints ;

        for (int i = 0; i < nz_TargetColumnMat.size(); i++){  //nz_TargetColumnMat[i].y = row index
            xCoordinatePoints.push_back(nz_TargetColumnMat[i].y);
        }

        std::sort(xCoordinatePoints.begin(), xCoordinatePoints.end()); //Sorting the vector

        int NassensteinDiameter=1;
        for (int j = xCoordinatePoints.size()-1; j >= 0; j--){
            if (xCoordinatePoints[j] - 1 == xCoordinatePoints[j-1]) NassensteinDiameter++;
            else break;
        }

        NassensteinDiameterAll.push_back(NassensteinDiameter);
    }

    //Now, Extract the statistics of the already computed features
    Statistics structStat;

    structStat = ComputeCommonStatistics (ChordAll);
    int min_Chord_All = structStat.min;
    int max_Chord_All = structStat.max;
    double mean_Chord_All = structStat.mean;
    double median_Chord_All = structStat.median;
    double std_Chord_All =  structStat.stdev;
    int mode_Chord_All = structStat.mode;
    ratios[31]=min_Chord_All;
    ratios[31]=max_Chord_All;
    ratios[31]=mean_Chord_All;
    ratios[31]=median_Chord_All;
    ratios[31]=std_Chord_All;
    ratios[31]=mode_Chord_All;

    structStat = ComputeCommonStatistics (LongestChordAll);
    int min_Longest_Chord = structStat.min;
    int max_Longest_Chord = structStat.max;
    double mean_Longest_Chord = structStat.mean;
    double median_Longest_Chord = structStat.median;
    double std_Longest_Chord =  structStat.stdev;
    int mode_Longest_Chord = structStat.mode;
    ratios[31]=min_Longest_Chord;
    ratios[31]=max_Longest_Chord;
    ratios[31]=mean_Longest_Chord;
    ratios[31]=median_Longest_Chord;
    ratios[31]=std_Longest_Chord;
    ratios[31]=mode_Longest_Chord;

    structStat = ComputeCommonStatistics (MartinLengthAll);
    int min_Martin_Length = structStat.min;
    int max_Martin_Length = structStat.max;
    double mean_Martin_Length = structStat.mean;
    double median_Martin_Length = structStat.median;
    double std_Martin_Length =  structStat.stdev;
    int mode_Martin_Length = structStat.mode;
    ratios[31]=min_Martin_Length;
    ratios[31]=max_Martin_Length;
    ratios[31]=mean_Martin_Length;
    ratios[31]=median_Martin_Length;
    ratios[31]=std_Martin_Length;
    ratios[31]=mode_Martin_Length;

    structStat = ComputeCommonStatistics (NassensteinDiameterAll);
    int min_Nassenstein_Diameter = structStat.min;
    int max_Nassenstein_Diameter = structStat.max;
    double mean_Nassenstein_Diameter = structStat.mean;
    double median_Nassenstein_Diameter = structStat.median;
    double std_Nassenstein_Diameter =  structStat.stdev;
    int mode_Nassenstein_Diameter = structStat.mode;
    ratios[31]=min_Nassenstein_Diameter;
    ratios[31]=max_Nassenstein_Diameter;
    ratios[31]=mean_Nassenstein_Diameter;
    ratios[31]=median_Nassenstein_Diameter;
    ratios[31]=std_Nassenstein_Diameter;
    ratios[31]=mode_Nassenstein_Diameter;

    //------------------------x_max and y_max------------------------------------------------------
    //https://git.rwth-aachen.de/ants/sensorlab/imea/-/blob/master/imea/measure_2d/macro.py#L304
    // x_max is the overall max chord of the object in all possible orientations
    // y_max is the longest chord orthogonal to `x_max`

    if (thetas.size()==0 || LongestChordAll.size()==0) std::cout<<"Error: Can not compute x_max and y_max as Input arrays are empty"<<std::endl;

    int x_max = *std::max_element(LongestChordAll.begin(), LongestChordAll.end());
    int idx_x_max = std::max_element(LongestChordAll.begin(),LongestChordAll.end()) - LongestChordAll.begin();

    int angle_x_max= thetas[idx_x_max];
    int angle_y_max = (angle_x_max + 90) % 180;

    std::vector<int>::iterator it = std::find(thetas.begin(), thetas.end(), angle_y_max);
    if (it == thetas.end())  std::cout << "Error: Cannot find theta element in computing x_max and y_max" << std::endl;

    int idx_y_max = std::distance(thetas.begin(), it);
    int y_max=LongestChordAll[idx_y_max];

    ratios[31]=x_max;
    ratios[31]=y_max;

    //--------------convexHull--------------------------------
    vector<vector<cv::Point>> contours;
    findContours( matrix, contours, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS ); //CHAIN_APPROX_TC89_KCOS had better performance when validated against MATLAB
    vector<vector<Point>> hull(1);
    vector<Point> AllPointsOnContours; // Make One vector from all the Points on the Contours of Various Objects

    for( size_t i = 0; i < contours.size(); i++ )
        for( size_t j = 0; j < contours[i].size(); j++ )
            AllPointsOnContours.push_back(contours[i][j]);

    convexHull(AllPointsOnContours, hull[0]);

    //Perimeter of the convex hull
    double convexHullPerimeter= arcLength(hull[0] , true);
    ratios[31]=convexHullPerimeter;


    //------------------------Erosion of the complement area between Convex Hul and the object------------------------------------------
    //Make the image for the complement area between Convex Hull and the object
    //First, fill the convex Hull interiors
    Mat ConvexHullFilledArea = Mat::zeros(matrix.rows, matrix.cols, CV_8UC1);
    drawContours( ConvexHullFilledArea, hull, -1, 255, FILLED);

    //Second, find the inverse area of the original matrix
    matrix.setTo(255, matrix==1);
    Mat matrix_inverse = Mat::zeros(matrix.rows, matrix.cols, CV_8UC1);
    bitwise_not(matrix, matrix_inverse);

    //Third, find the logical AND area between the two matrix above
    Mat finalComplementImage = Mat::zeros(matrix.rows, matrix.cols, CV_8UC1);
    bitwise_and(matrix_inverse, ConvexHullFilledArea, finalComplementImage);
   // imwrite("1.tiff", finalComplementImage);

    vector <int> Pixels_erosion_Complement; //Vector to store pixel counts at each erosion iteration
    vector<Point> finalComplementImage_nonZero;
    findNonZero(finalComplementImage , finalComplementImage_nonZero);
    Pixels_erosion_Complement.push_back(finalComplementImage_nonZero.size());

    Mat erosion_dst_Complement;
    vector<Point> erosion_dst_nonZero_Complement;
    Mat element_Complement = Mat();

    //Continue the erosion process until the area vanishes
    while (Pixels_erosion_Complement[Pixels_erosion_Complement.size()-1] !=0 ){
        erode( finalComplementImage, erosion_dst_Complement, element_Complement); //Apply erosion on the binary image

        findNonZero(erosion_dst_Complement , erosion_dst_nonZero_Complement);  //count number of white pixels
        Pixels_erosion_Complement.push_back(erosion_dst_nonZero_Complement.size()); //Store the counts of the white pixels

        //erosion_dst_Complement.setTo(255, erosion_dst_Complement==1);
        //string filename = to_string(Pixels_erosion_Complement.size())  +".tiff";
        //imwrite(filename, erosion_dst_Complement);
        finalComplementImage=erosion_dst_Complement;
    }

    ratios[31]=Pixels_erosion_Complement.size();


    //------------------------Erosion of the object------------------------------------------

    vector <int> Pixels_erosion; //Vector to store pixel counts at each erosion iteration

    vector<Point> matrix_nonZero;
    findNonZero(matrix , matrix_nonZero);
    Pixels_erosion.push_back(matrix_nonZero.size());

    Mat erosion_dst;
    vector<Point> erosion_dst_nonZero;

    Mat element = Mat();

    //matrix.setTo(255, matrix==1);
    //imwrite("1.tiff", matrix);
    //Continue the erosion process until the object vanishes
    while (Pixels_erosion[Pixels_erosion.size()-1] !=0 ){
        erode( matrix, erosion_dst, element);//Apply erosion on the binary image

        findNonZero(erosion_dst , erosion_dst_nonZero); //count number of white pixels
        Pixels_erosion.push_back(erosion_dst_nonZero.size()); //Store the counts of the white pixels

       // erosion_dst.setTo(255, erosion_dst==1);
       // int counter=Pixels_erosion.size();
       // string filename = to_string(counter)  +".tiff";
       // imwrite(filename, erosion_dst);

        matrix=erosion_dst;
    }

    ratios[31]=Pixels_erosion.size();

    //------------------------Minimum enclosing circle------------------------------------------
    // Find the minimum enclosing circle of an object
    Point2f center;
    float radius = 0;
    minEnclosingCircle(AllPointsOnContours, center, radius);
    //Diameter of the minimum circumference of the projection area.
    double diameter_min_enclosing_circle = 2*radius;
    ratios[30]=diameter_min_enclosing_circle;

    //--------------Rotated Bounding Box--------------------------------
    //Computing Rotated Bounding Box
    //https://git.rwth-aachen.de/ants/sensorlab/imea/-/blob/master/imea/measure_2d/macro.py#L44
    RotatedRect rotatedBoundingBox= minAreaRect(AllPointsOnContours);

    //For Consistency with imea, we will only report Height and Width of the Bounding Box
    //and the other following parameters are only being reported for the sake of completeness
    double BBHeight=rotatedBoundingBox.size.width+1;
    double BBWidth=rotatedBoundingBox.size.height+1;
    ratios[31]=BBHeight;
    ratios[31]=BBWidth;

    //Followings are optional
    double BufferCorrectionY= Im.ROIHeightBegActual - Im.ROIHeightBeg;
    double BufferCorrectionX= Im.ROIWidthBegActual - Im.ROIWidthBeg;

    double CenterHeight= rotatedBoundingBox.center.y+0.5-BufferCorrectionY;
    double CenterWidth= rotatedBoundingBox.center.x+0.5-BufferCorrectionX;

    //Coordinates of the 4 points of the rotated bounding box
    vector<Point2f> boxPts(4);
    rotatedBoundingBox.points(boxPts.data());

    double P1Height=boxPts[0].y-BufferCorrectionY;
    double P1Width=boxPts[0].x-BufferCorrectionX;
    double P2Height=boxPts[1].y-BufferCorrectionY;
    double P2Width=boxPts[1].x+1-BufferCorrectionX;
    double P3Height=boxPts[2].y+1-BufferCorrectionY;
    double P3Width=boxPts[2].x+1-BufferCorrectionX;
    double P4Height=boxPts[3].y+1-BufferCorrectionY;
    double P4Width=boxPts[3].x-BufferCorrectionX;
    //------------------------Euler Number------------------------------------------
    long Euler= EulerNumber(arr,8,Im.height,Im.width);
    ratios[24]=Euler;
    delete [] arr;

    //------------------------------Some Other Statistics--------
    double extent = (float)PixelsCount/(float)boundingBoxArea;
    double convexHullArea = contourArea(hull[0]);
    double solidity = PixelsCount/convexHullArea;

    double AspectRatio=(float)Im.ROIWidthActual/(float)Im.ROIHeightActual;

    //Diameter of a circle of equal area
    double EquivalentDiameter= sqrt(4/M_PI*PixelsCount);
    double ROIPerimeter=0;

    for( size_t i = 0; i < contours.size(); i++ )
        ROIPerimeter += arcLength(contours[i],true);

    double Circularity=4*M_PI*PixelsCount/(ROIPerimeter*ROIPerimeter);

    ratios[25]=extent;
    ratios[26]=convexHullArea;
    ratios[27]=solidity;
    ratios[28]=AspectRatio;
    ratios[29]=EquivalentDiameter;
    ratios[30]=ROIPerimeter;
    ratios[31]=Circularity;  

    //Diameter of a circle of equal perimeter
    double diameter_equal_perimeter=ROIPerimeter/M_PI;
    ratios[31]=diameter_equal_perimeter;

    //------------------------------Geodetic Length and Thickness------------
    /*
    * Accroding to imea code: https://git.rwth-aachen.de/ants/sensorlab/imea/-/blob/master/imea/measure_2d/macro.py#L244
    * the definitions of Geodetic Length and Thickness are as follows.
    The geodetic lengths and thickness are approximated by a rectangle
    with the same area and perimeter:
    (1) `area = geodeticlength * thickness`
    (2) `perimeter = 2 * (geodetic_length + thickness)`

    # White the help of Equation (1) we can rewrite Equation (2) to:
    # `geodetic_length**2 - 0.5 * perimeter * geodetic_length + area = 0`
    # Which we can solve with the pq-formula:
    # `geodetic_length = perimeter/4 +- sqrt((perimeter/4)**2 - area)`
    # since only the positive solution makes sense in our application
    */

    if (ROIPerimeter <= 0) cout<<" Perimeter should be a positive value greater than zero"<<endl;
    if (PixelsCount <= 0) cout<<" Area should be a positive value greater than zero"<<endl;

    double SqRootTmp = ROIPerimeter*ROIPerimeter/16 - (double)PixelsCount;

    //Make sure value under SqRootTmp is always positive
    if (SqRootTmp <0) SqRootTmp=0;

    //Calcuate geodetic_length with pq-formula (see above):
    double geodetic_length = ROIPerimeter/4 + sqrt(SqRootTmp);

    //Calculate thickness by rewriting Equation (2):
    double thickness = ROIPerimeter/2 - geodetic_length;

    ratios[30]=geodetic_length;
    ratios[31]=thickness;

    //---------------------Min/Max Feret Diameter/Angle-----------------------
    double * MaxDistanceArray = new double [180];
    vector <double> FeretDiameterAll;

    for (int i=0; i<180; ++i){
        float theta=i*M_PI/180;
        double MaxXCoord=-INF;
        double MinXCoord=INF;
        int MaxIndex=-1;
        int MinIndex=-1;

        for (int j = 0; j < hull[0].size(); j++) {
            float rotatedX = std::cos(theta)*(hull[0][j].x) - std::sin(theta)*(hull[0][j].y);
            if (rotatedX > MaxXCoord) {MaxXCoord = rotatedX; MaxIndex=j;}
            if (rotatedX < MinXCoord) {MinXCoord = rotatedX; MinIndex=j;}
        }

        if (MaxIndex == -1 || MinIndex == -1) cout<< "Something Went Wrong!"<<endl;

        //1 (2x0.5) was added in the below line for consistency with MATLAB as the side of the pixel is 0.5 off from the center.
        MaxDistanceArray[i]=  MaxXCoord-MinXCoord+1;
        FeretDiameterAll.push_back(MaxXCoord-MinXCoord+1);
    }

    double MaxFeretDiameter=0;
    double MinFeretDiameter=INF;
    int MaxFeretAngle=-1;
    int MinFeretAngle=-1;

    for (int i=0; i<180; ++i){
        if (MaxDistanceArray[i] > MaxFeretDiameter) {MaxFeretDiameter = MaxDistanceArray[i]; MaxFeretAngle=i;}
        if (MaxDistanceArray[i] < MinFeretDiameter) {MinFeretDiameter = MaxDistanceArray[i]; MinFeretAngle=i;}
    }

    ratios[32]=MaxFeretDiameter;
    ratios[33]=MaxFeretAngle; //The angle is between 0 to 180. MATLAB reports -180 to 180 instead.

    ratios[34]=MinFeretDiameter;
    ratios[35]=MinFeretAngle; //The angle is between 0 to 180. MATLAB reports -180 to 180 instead.

    delete [] MaxDistanceArray;

    //Now, compute Feret Statistics
    structStat = ComputeCommonStatistics2 (FeretDiameterAll);
    int min_Feret_Diameter = structStat.min;
    int max_Feret_Diameter = structStat.max;
    double mean_Feret_Diameter = structStat.mean;
    double median_Feret_Diameter = structStat.median;
    double std_Feret_Diameter =  structStat.stdev;
    int mode_Feret_Diameter = structStat.mode;
    ratios[31]=min_Feret_Diameter;
    ratios[31]=max_Feret_Diameter;
    ratios[31]=mean_Feret_Diameter;
    ratios[31]=median_Feret_Diameter;
    ratios[31]=std_Feret_Diameter;
    ratios[31]=mode_Feret_Diameter;

    //----------------------Finding Neighbors for the Current ROI-----------------
    uint32_t imageWidth, imageLength;
    imageWidth=Im.ROIWidthEndLabel-Im.ROIWidthBegLabel+1;
    imageLength=Im.ROIHeightEndLabel-Im.ROIHeightBegLabel+1;

    int  PixelDistance=5;  //Search Distance from each side

    int buffery=Im.ROIHeightBeg-Im.ROIHeightBegLabel;
    int bufferx=Im.ROIWidthBeg-Im.ROIWidthBegLabel;
    vector<int> NeighborIDs;

    for (int i = 0; i < contours.size(); i++) {
        Point sampleBorderPoint = contours[i][0];

        for (int j = 0; j < contours[i].size(); j++) {
            Point borderPoint = contours[i][j];
            int Px=borderPoint.x+bufferx;  //Translate coordinates from IntensityImage BoundingBox to Labeled Image BoundingBox
            int Py=borderPoint.y+buffery;  //Translate coordinates from IntensityImage BoundingBox to Labeled Image BoundingBox

           for (int l=Py-PixelDistance; l<Py+PixelDistance+1; ++l)
               for (int k=Px-PixelDistance; k<Px+PixelDistance+1; ++k){

                   if (k<0 || k>imageWidth-1) continue;
                   if (l<0 || l>imageLength-1) continue;

                   int value= Im.ROI_Bounding_Box_Labels[l][k];

                   if (value!=0 && value!=Im.LabelID)  NeighborIDs.push_back(value);
               }
       }
   }

    sort(NeighborIDs.begin(),NeighborIDs.end());

    // Remove duplicates (v1)
    std::vector<int> NeighborUniqueIDs;
    std::unique_copy(NeighborIDs.begin(), NeighborIDs.end(), std::back_inserter(NeighborUniqueIDs));

    ratios[36]=NeighborUniqueIDs.size();

    //--------------------------------Hexagonality/Polygonality-------------------
    //This section is a translation from the following Python code
    //https://github.com/LabShare/polus-plugins/blob/master/polus-feature-extraction-plugin/src/main.py

    int neighbors=NeighborUniqueIDs.size();
    double area=PixelsCount;
    double perimeter = ROIPerimeter;
    double area_hull=convexHullArea;
    double perim_hull=6*sqrt(area_hull/(1.5*sqrt(3)));
    double perimeter_neighbors;

    if (neighbors ==0) perimeter_neighbors=std::numeric_limits<double>::quiet_NaN();
    else if (neighbors > 0) perimeter_neighbors= perimeter/neighbors;

    //Polygonality metrics calculated based on the number of sides of the polygon
    if (neighbors > 2){

        double poly_size_ratio = 1.0 - abs(1.0 - perimeter_neighbors/sqrt((4*area)/(neighbors/tan(M_PI/neighbors))));
        double poly_area_ratio = 1.0 - abs(1.0 - area/(0.25*neighbors* perimeter_neighbors* perimeter_neighbors / tan(M_PI/neighbors)));

        //Calculate Polygonality Score
        double poly_ave = 10 * (poly_size_ratio+poly_area_ratio)/2;

        //Hexagonality metrics calculated based on a convex, regular, hexagon
        double apoth1 = sqrt(3)*perimeter/12;
        double apoth2 = sqrt(3)*MaxFeretDiameter/4;
        double apoth3 = MinFeretDiameter/2;
        double side1 = perimeter/6;
        double side2 = MaxFeretDiameter/2;
        double side3 = MinFeretDiameter/sqrt(3);
        double side4 = perim_hull/6;

        //Unique area calculations from the derived and primary measures above
        double area1 = 0.5*(3*sqrt(3))*side1*side1;
        double area2 = 0.5*(3*sqrt(3))*side2*side2;
        double area3 = 0.5*(3*sqrt(3))*side3*side3;
        double area4 = 3*side1*apoth2;
        double area5 = 3*side1*apoth3;
        double area6 = 3*side2*apoth3;
        double area7 = 3*side4*apoth1;
        double area8 = 3*side4*apoth2;
        double area9 = 3*side4*apoth3;
        double area10= area_hull;
        double area11= area;

        //Create an array of all unique areas
        vector<double> list_area={area1,area2,area3,area4,area5,area6,area7,area8,area9,area10,area11};
        vector<double> area_array;

        //Create Summary statistics of all array ratios
        double sum=0;
        for (int ib=0; ib<list_area.size(); ++ib)
            for (int ic=ib+1; ic<list_area.size(); ++ic) {
                double area_ratio=1.0- abs(1.0 - list_area[ib]/list_area[ic]);
                area_array.push_back(area_ratio);
                sum+=area_ratio;
            }
        double area_ratio_ave=sum/area_array.size();

        double sqrdTmp=0;
        for (int i=0; i<area_array.size(); ++i){
            sqrdTmp += (area_array[i]- area_ratio_ave)*(area_array[i]- area_ratio_ave);
        }
        double area_ratio_sd=sqrt(sqrdTmp/area_array.size());

        //Set the hexagon area ratio equal to the average Area Ratio
        double hex_area_ratio = area_ratio_ave;

        //Perimeter Ratio Calculations
        //Two extra apothems are now useful
        double apoth4 = sqrt(3) * perim_hull / 12;
        double apoth5 = sqrt(4 * area_hull / (4.5 * sqrt(3)));
        double perim1 = sqrt(24 * area / sqrt(3));
        double perim2 = sqrt(24 * area_hull / sqrt(3));
        double perim3 = perimeter;
        double perim4 = perim_hull;
        double perim5 = 3 * MaxFeretDiameter;
        double perim6 = 6 * MinFeretDiameter / sqrt(3);
        double perim7 = 2 * area / apoth1;
        double perim8 = 2 * area / apoth2;
        double perim9 = 2 * area / apoth3;
        double perim10 = 2 * area / apoth4;
        double perim11 = 2 * area / apoth5;
        double perim12 = 2 * area_hull / apoth1;
        double perim13 = 2 * area_hull / apoth2;
        double perim14 = 2 * area_hull / apoth3;

        //Create an array of all unique Perimeters
        vector<double> list_perim = {perim1,perim2,perim3,perim4,perim5,perim6,perim7,perim8,perim9,perim10,perim11,perim12,perim13,perim14};
        vector<double> perim_array;

        //Create an array of the ratio of all Perimeters to eachother
        //Create Summary statistics of all array ratios
        double sum2=0;
        for (int ib=0; ib<list_perim.size(); ++ib)
            for (int ic=ib+1; ic<list_perim.size(); ++ic) {
                double perim_ratio=1.0- abs(1.0 - list_perim[ib]/list_perim[ic]);
                perim_array.push_back(perim_ratio);
                sum2+=perim_ratio;
            }
        double perim_ratio_ave=sum2/perim_array.size();

        double sqrdTmp2=0;
        for (int i=0; i<perim_array.size(); ++i){
            sqrdTmp2 += (perim_array[i]- perim_ratio_ave)*(perim_array[i]- perim_ratio_ave);
        }
        double perim_ratio_sd=sqrt(sqrdTmp2/perim_array.size());

        //Set the HSR equal to the average Perimeter Ratio
        double hex_size_ratio = perim_ratio_ave;
        double hex_sd=sqrt((area_ratio_sd*area_ratio_sd+perim_ratio_sd*perim_ratio_sd)/2);
        double hex_ave = 10*(hex_area_ratio + hex_size_ratio)/2;

        ratios[37]=poly_ave;
        ratios[38]=hex_ave;
        ratios[39]=hex_sd;

    }
    else if (neighbors <3 ){
        double poly_ave = std::numeric_limits<double>::quiet_NaN();
        double hex_ave = std::numeric_limits<double>::quiet_NaN();
        double hex_sd = std::numeric_limits<double>::quiet_NaN();

        ratios[37]=poly_ave;
        ratios[38]=hex_ave;
        ratios[39]=hex_sd;
    }

    //  cout << "Finished Morphological Computations!\n" << endl;

        //Now Lets Rearrange outputs to match Jayapriya's Python Code
            vector<double> tmp2;
            tmp2.push_back((double)PixelsCount*UnitConversion*UnitConversion); //Area
            tmp2.push_back(yCentroid*UnitConversion);
            tmp2.push_back(xCentroid*UnitConversion);
            tmp2.push_back(BoundingBoxHeightBeg*UnitConversion);
            tmp2.push_back(BoundingBoxWidthBeg*UnitConversion);
            tmp2.push_back(Im.ROIHeightActual*UnitConversion);
            tmp2.push_back(Im.ROIWidthActual*UnitConversion);
            tmp2.push_back(MajorAxisLength*UnitConversion);
            tmp2.push_back(MinorAxisLength*UnitConversion);
            tmp2.push_back(Eccentricity);
            tmp2.push_back(Orientation);
            tmp2.push_back(convexHullArea*UnitConversion*UnitConversion);
            tmp2.push_back(Euler);
            tmp2.push_back(EquivalentDiameter*UnitConversion);
            tmp2.push_back(solidity);
            tmp2.push_back(ROIPerimeter*UnitConversion);
            tmp2.push_back(MaxFeretDiameter*UnitConversion);
            tmp2.push_back(MinFeretDiameter*UnitConversion);
            tmp2.push_back(NeighborUniqueIDs.size());
            tmp2.push_back(ratios[37]); //poly_ave
            tmp2.push_back(ratios[38]); //hex_ave
            tmp2.push_back(ratios[39]); //hex_sd

            tmp2.push_back(Circularity);  //MATLAB Features start here
            tmp2.push_back(ratios[8]*UnitConversion); //Extrema begins here
            tmp2.push_back(ratios[16]*UnitConversion);
            tmp2.push_back(ratios[9]*UnitConversion);
            tmp2.push_back(ratios[17]*UnitConversion);
            tmp2.push_back(ratios[10]*UnitConversion);
            tmp2.push_back(ratios[18]*UnitConversion);
            tmp2.push_back(ratios[11]*UnitConversion);
            tmp2.push_back(ratios[19]*UnitConversion);
            tmp2.push_back(ratios[12]*UnitConversion);
            tmp2.push_back(ratios[20]*UnitConversion);
            tmp2.push_back(ratios[13]*UnitConversion);
            tmp2.push_back(ratios[21]*UnitConversion);
            tmp2.push_back(ratios[14]*UnitConversion);
            tmp2.push_back(ratios[22]*UnitConversion);
            tmp2.push_back(ratios[15]*UnitConversion);
            tmp2.push_back(ratios[23]*UnitConversion); //Extrema ends here
            tmp2.push_back(extent);
            tmp2.push_back(MaxFeretAngle);  //No MaxFeret Coordinates
            tmp2.push_back(MinFeretAngle);  //No MinFeret Coordinates

            for (int i=0; i<tmp2.size(); ++i){
                ratios[i] = tmp2[i];
            }

    return;
}










