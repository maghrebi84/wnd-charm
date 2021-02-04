
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* Written by:                                                                   */
/*        Mahdi Maghrebi <mahdi.maghrebi [at] nih [dot] gov>                     */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/

#include "MorphologicalAlgorithms.h"
#include "cmatrix.h"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <vector>
#include <math.h>
#include <boost/filesystem.hpp>
#include "cmatrix.h"
#include <memory>
#include <vector>
#include <ome/files/in/OMETIFFReader.h>
#include <ome/files/FormatReader.h>
#include <ome/files/in/TIFFReader.h>
#include <ome/files/tiff/IFD.h>
#include <tiffio.h>

#include <fstream>;  //??

using ome::files::dimension_size_type;
using ome::files::FormatReader;
using ome::files::MetadataMap;
using ome::files::VariantPixelBuffer;
using ome::files::PixelBuffer;
using namespace std;
using namespace cv;


static void readOriginalMetadata2(const FormatReader& reader, std::ostream& stream, int tileType, uint32_t& tileWidth, uint32_t& tileLength,
                                  uint32_t& imageWidth, uint32_t& imageLength,uint32_t& bitsPerSample,uint16_t& samplesPerPixel)
{
    // Get total number of images (series)
    dimension_size_type ic = reader.getSeriesCount();

    // Get global metadata
    const MetadataMap& global = reader.getGlobalMetadata();

    // Loop over images
    for (dimension_size_type i = 0 ; i < ic; ++i)
    {
        // Change the current series to this index
        reader.setSeries(i);

        // Print series metadata
        const MetadataMap& series = reader.getSeriesMetadata();
        const MetadataMap::key_type imageWidthKey ="ImageWidth";
        const MetadataMap::key_type imageLengthKey ="ImageLength";
        const MetadataMap::key_type bitsPerSampleKey ="BitsPerSample";
        const MetadataMap::key_type samplesPerPixelKey ="SamplesPerPixel";

        imageWidth = ome::compat::get<uint32_t>(reader.getSeriesMetadataValue(imageWidthKey));
        imageLength = ome::compat::get<uint32_t>(reader.getSeriesMetadataValue(imageLengthKey));
        bitsPerSample = ome::compat::get<uint32_t>(reader.getSeriesMetadataValue(bitsPerSampleKey));
        samplesPerPixel= ome::compat::get<uint16_t>(reader.getSeriesMetadataValue(samplesPerPixelKey));

        if (tileType==1){
            const MetadataMap::key_type tileWidthKey ="TileWidth";
            const MetadataMap::key_type tileLengthKey ="TileLength";
            tileWidth = ome::compat::get<uint32_t>(reader.getSeriesMetadataValue(tileWidthKey));
            tileLength = ome::compat::get<uint32_t>(reader.getSeriesMetadataValue(tileLengthKey));
        }

    }
}

struct Visitor2
{
    //    double is used since it can contain the value for any pixel type
    typedef std::vector<double> result_type;
    result_type myvec;

    // Get min and max for any non-complex pixel type
    template<typename T>
    result_type operator() (const T& v)
    {
        typedef typename T::element_type::value_type value_type;

        for (auto i=0; i<v->num_elements();i++){
            value_type tmp = v->data()[i];
            myvec.push_back(static_cast<double>(tmp));
        }
        return myvec;
    }
    //----------------------------------------------------------------------------------------------
    //The rest was kept from the OME online example as it is necessary for compilation
    //However, functionality to read complex pixel values are not implemented and is left for future
    //----------------------------------------------------------------------------------------------
    // Less than comparison for real part of complex numbers
    template <typename T>
    static bool
    complex_real_less(const T& lhs, const T& rhs)
    {
        return std::real(lhs) < std::real(rhs);
    }

    // Greater than comparison for real part of complex numbers
    template <typename T>
    static bool
    complex_real_greater(const T& lhs, const T& rhs)
    {
        return std::real(lhs) > std::real(rhs);
    }

    // This is the same as for simple pixel types, except for the
    // addition of custom comparison functions and conversion of the
    // result to the real part.
    template <typename T>
    typename boost::enable_if_c<
    boost::is_complex<T>::value, result_type
    >::type
    operator() (const std::shared_ptr<PixelBuffer<T>>& v)
    {
        typedef T value_type;

        value_type *min = std::min_element(v->data(),
                                           v->data() + v->num_elements(),
                                           complex_real_less<T>);
        value_type *max = std::max_element(v->data(),
                                           v->data() + v->num_elements(),
                                           complex_real_greater<T>);

        myvec.push_back(static_cast<double>(std::real(*min)));
        myvec.push_back(static_cast<double>(std::real(*max)));

        return myvec;
    }
};


long EulerNumber(unsigned char ** pix_plane, int mode, int height, int width) {
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
            if (pix_plane[y-1][x-1] > 0) Imq |=  (1 << 3);
            if (pix_plane[y-1][x] > 0) Imq |=  (1 << 2);
            if (pix_plane[y][x-1] > 0) Imq |=  (1 << 1);
            if (pix_plane[y][x] > 0) Imq |=  (1 << 0);
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

/* the input should be a binary image */
void WeightedGlobalCentroid(const ImageMatrix &Im, double * x_centroid, double * y_centroid) {
    unsigned int x,y,w = Im.width, h = Im.height;
    double x_mass=0,y_mass=0,mass=0;
    readOnlyPixels pix_plane = Im.ReadablePixels();

    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            if (!std::isnan(pix_plane(y,x))){
                x_mass=x_mass+(x+1)*pix_plane(y,x);    /* the "+1" is only for compatability with matlab code (where index starts from 1) */
                y_mass=y_mass+(y+1)*pix_plane(y,x);    /* the "+1" is only for compatability with matlab code (where index starts from 1) */
                mass+=pix_plane(y,x);
            }
    if (mass) {
        *x_centroid=x_mass/mass+Im.ROIWidthBeg;
        *y_centroid=y_mass/mass+Im.ROIHeightBeg;
    } else *x_centroid=*y_centroid=0;
}


double** readLabeledImage(char* ROIPath, uint32_t * imageWidth0, uint32_t * imageLength0) {

    shared_ptr<ome::files::FormatReader> reader(std::make_shared<ome::files::in::TIFFReader>());

    // Set reader options before opening a file
    reader->setMetadataFiltered(false);
    reader->setGroupFiles(true);

    // Open the file
    reader->setId(ROIPath);

    // Display series core metadata
    //readMetadata2(*reader, std::cout);

    std::shared_ptr<ome::files::tiff::TIFF> myTiff = ome::files::tiff::TIFF::open(ROIPath, "r");
    std::shared_ptr<ome::files::tiff::IFD> ifd (myTiff->getDirectoryByIndex(0));
    int tileType=ifd->getTileInfo().tileType();
    int tileCount=ifd->getTileInfo().tileCount();

    uint32_t tileWidth, tileLength, bitsPerSample;
    uint32_t imageWidth, imageLength;
    uint16_t samplesPerPixel;

    // Display global and series original metadata
    readOriginalMetadata2(*reader, std::cout, tileType, tileWidth, tileLength, imageWidth, imageLength, bitsPerSample,samplesPerPixel);

    *imageWidth0=imageWidth;
    *imageLength0=imageLength;

    // Get total number of images (series)
    dimension_size_type ic = reader->getSeriesCount();

    reader->setSeries(0);
    VariantPixelBuffer buf;

    double ** LabeledImage;
    LabeledImage = new double*[imageLength];  //Convert double to int ????
    for (int i = 0; i < imageLength; ++i) { LabeledImage[i] = new double[imageWidth]; }

    unsigned int h,w,x=0,y=0;
    unsigned short int spp=0,bps=0;
    unsigned int width=imageWidth;
    unsigned int height=imageLength;
    unsigned short int bits=bitsPerSample;
    spp= samplesPerPixel;

    //if ( ! (bits == 8 || bits == 16) ) return (0); // only 8 and 16-bit images supported.  //??error message here
    if (!spp) spp=1;  /* assume one sample per pixel if nothing is specified */
    //if (spp >1) {  printf(" spp is bigger than 1 ");return -1;}

    if (tileType==1){
        for (y = 0; y < height; y += tileLength) {
            for (x = 0; x < width; x += tileWidth) {
                reader->openBytes(0, buf,x,y,tileWidth,tileLength);

                Visitor2 visitor;
                Visitor2::result_type result = ome::compat::visit(visitor, buf.vbuffer());

                int rowMax = std::min(y + tileLength, height);
                int colMax = std::min(x + tileWidth, width);

                for (unsigned int rowtile = 0; rowtile < rowMax - y; ++rowtile) {
                    int col,count=0;
                    unsigned int coltile=0;

                    while (coltile<colMax - x) {
                        double val = result[rowtile*tileWidth+coltile];
                        LabeledImage[y+rowtile][x+coltile] = val;
                        coltile++;
                    }
                }
            }
        }
        // Explicitly close reader
        reader->close();
    }
    else {
        printf(" The format of Labeled Image is not tiled-tiff ");
    }
    return LabeledImage;
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

    ratios[70]=P1x;
    ratios[71]=P2x;
    ratios[72]=P3x;
    ratios[73]=P4x;
    ratios[74]=P5x;
    ratios[75]=P6x;
    ratios[76]=P7x;
    ratios[77]=P8x;
    ratios[78]=P1y;
    ratios[79]=P2y;
    ratios[80]=P3y;
    ratios[81]=P4y;
    ratios[82]=P5y;
    ratios[83]=P6y;
    ratios[84]=P7y;
    ratios[85]=P8y;

    return;
}


void MorphologicalAlgorithms(const ImageMatrix &Im, double *ratios){

    //----------Total Number of ROI Pixels--------------
    int PixelsCount=Im.stats.n();

    ratios[0]=(double)PixelsCount;

    //---------------Position and size of the smallest box containing the region (Bounding Box)---------
    double BoundingBoxWidthBeg=Im.ROIWidthBegActual+1-0.5;  //0.5 was added to make the results consistent with MATLAB
    double BoundingBoxHeightBeg=Im.ROIHeightBegActual+1-0.5; //0.5 was added to make the results consistent with MATLAB
    int boundingBoxArea=Im.ROIWidthActual*Im.ROIHeightActual;

    ratios[1]=BoundingBoxWidthBeg;
    ratios[2]=BoundingBoxHeightBeg;
    ratios[3]=Im.ROIWidthActual;
    ratios[4]=Im.ROIHeightActual;
    ratios[5]=boundingBoxArea;

    //--------------Spatial Centroids----------------
    double xCentroid,yCentroid;
    GlobalCentroid2(Im,&xCentroid,&yCentroid);
    ratios[5]=xCentroid;
    ratios[6]=yCentroid;

    //Weighted Centroid
    double xWCentroid,yWCentroid;
    WeightedGlobalCentroid(Im,&xWCentroid,&yWCentroid);
    ratios[50]=xWCentroid;
    ratios[51]=yWCentroid;

    //--------------Statistics and Moments-----------
    double mean= Im.stats.mean();
    double max= Im.stats.max();
    double min= Im.stats.min();
    double median=Im.get_median();

    uchar* arr = new uchar[Im.height*Im.width];

    readOnlyPixels in_plane = Im.ReadablePixels();

    double SqrdTmp=0;
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
                SqrdTmp += tmp*tmp;
                TrpdTmp += tmp*tmp*tmp;
                QuadTmp += tmp*tmp*tmp*tmp;
            }
        }

    double Variance= SqrdTmp/Im.stats.n();
    double STDEV= sqrt(Variance);
    double Skewness= (TrpdTmp/Im.stats.n())/pow(STDEV,3);
    double Kurtosis= (QuadTmp/Im.stats.n())/pow(Variance,2) - 3;

    ratios[100]=mean;
    ratios[101]=min;
    ratios[102]=max;
    ratios[103]=median;
    ratios[104]=STDEV;
    ratios[105]=Skewness;
    ratios[106]=Kurtosis;

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
-
    //--------------Coordinates of the extreme Pixels--------
    Extrema (Im, ratios);

    //--------------convexHull--------------------------------

    cv::Mat matrix = cv::Mat(Im.height,Im.width,CV_8UC1,arr);

    if( matrix.empty() ) {
        cout << "Could not find the image!\n" << endl;
        return ;
    }

    vector<vector<Point>> contours;
    findContours( matrix, contours, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS ); //CHAIN_APPROX_TC89_KCOS had better performance when validated against MATLAB
    vector<vector<Point>> hull(1);
    vector<Point> AllPointsOnContours; // Make One vector from all the Points on the Contours of Various Objects

    for( size_t i = 0; i < contours.size(); i++ )
        for( size_t j = 0; j < contours[i].size(); j++ )
            AllPointsOnContours.push_back(contours[i][j]);

    convexHull(AllPointsOnContours, hull[0]);
    delete [] arr;

    //--------------------------------------

    double extent = (float)PixelsCount/(float)boundingBoxArea;
    double convexHullArea = contourArea(hull[0]);
    double solidity = PixelsCount/convexHullArea;

    double AspectRatio=(float)Im.ROIWidthActual/(float)Im.ROIHeightActual;

    double EquivalentDiameter= sqrt(4/M_PI*PixelsCount);
    double ROIPerimeter=0;

    for( size_t i = 0; i < contours.size(); i++ )
            ROIPerimeter += arcLength(contours[i],true);

    double Circularity=4*M_PI*PixelsCount/(ROIPerimeter*ROIPerimeter);

    ratios[71]=extent;
    ratios[71]=convexHullArea;
    ratios[71]=solidity;
    ratios[71]=AspectRatio;
    ratios[71]=EquivalentDiameter;
    ratios[71]=ROIPerimeter;
    ratios[71]=Circularity;
    //---------------------Min/Max Feret Diameter/Angle-----------------------

    double * MaxDistanceArray = new double [180];
    int * Point1Index = new int [180];
    int * Point2Index = new int [180];

    for (int i=0; i<180; ++i){

        float theta=i*M_PI/180;
        double MaxXCoord=-INF;
        double MinXCoord=INF;
        int MaxIndex=-1;
        int MinIndex=-1;

        for (int j = 0; j < hull[0].size(); j++) {
            // Point coordinate_i_ofcontour = hull[0][j];
            //outPoint.x = std::cos(angRad)*inPoint.x - std::sin(angRad)*inPoint.y;
            //outPoint.y = std::sin(angRad)*inPoint.x + std::cos(angRad)*inPoint.y;

            float rotatedX = std::cos(theta)*(hull[0][j].x) - std::sin(theta)*(hull[0][j].y);
            if (rotatedX > MaxXCoord) {MaxXCoord = rotatedX; MaxIndex=j;}
            if (rotatedX < MinXCoord) {MinXCoord = rotatedX; MinIndex=j;}
        }

        if (MaxIndex == -1 || MinIndex == -1) cout<< "Something Went Wrong!"<<endl;

        MaxDistanceArray[i]=  MaxXCoord-MinXCoord+1; //1 for consistency with MATLAB
        Point1Index[i]= MinIndex;
        Point2Index[i]= MaxIndex;
    }

    double MaxFeretDiameter=0;
    double MinFeretDiameter=INF;
    int MaxFeretIndex=-1;
    int MinFeretIndex=-1;

    for (int i=0; i<180; ++i){
        if (MaxDistanceArray[i] > MaxFeretDiameter) {MaxFeretDiameter = MaxDistanceArray[i]; MaxFeretIndex=i;}
        if (MaxDistanceArray[i] < MinFeretDiameter) {MinFeretDiameter = MaxDistanceArray[i]; MinFeretIndex=i;}
    }

    ratios[200]=MaxFeretDiameter;

    ratios[202]=hull[0][Point2Index[MaxFeretIndex]].x+0.5+Im.ROIWidthBeg;  //0.5 was added for consistency with MATLAB
    ratios[203]=hull[0][Point1Index[MaxFeretIndex]].x+0.5+Im.ROIWidthBeg;  //0.5 was added for consistency with MATLAB
    ratios[204]=hull[0][Point2Index[MaxFeretIndex]].y+0.5+Im.ROIHeightBeg;  //0.5 was added for consistency with MATLAB
    ratios[205]=hull[0][Point1Index[MaxFeretIndex]].y+0.5+Im.ROIHeightBeg;  //0.5 was added for consistency with MATLAB

    ratios[201]=180/M_PI*atan((ratios[204]-ratios[205])/(ratios[202]-ratios[203]));
    if (ratios[201] < 0) ratios[201]=180+ratios[201];


    ratios[206]=MinFeretDiameter;

    ratios[208]=hull[0][Point2Index[MinFeretIndex]].x+0.5+Im.ROIWidthBeg;   //0.5 was added for consistency with MATLAB
    ratios[209]=hull[0][Point1Index[MinFeretIndex]].x+0.5+Im.ROIWidthBeg;   //0.5 was added for consistency with MATLAB
    ratios[210]=hull[0][Point2Index[MinFeretIndex]].y+0.5+Im.ROIHeightBeg;   //0.5 was added for consistency with MATLAB
    ratios[211]=hull[0][Point1Index[MinFeretIndex]].y+0.5+Im.ROIHeightBeg;   //0.5 was added for consistency with MATLAB

    ratios[207]=180/M_PI*atan((ratios[210]-ratios[211])/(ratios[208]-ratios[209]));
    if (ratios[207] < 0) ratios[207]=180+ratios[207];

    delete [] MaxDistanceArray, Point1Index, Point2Index;


    //----------------------Finding Neighbors for the Current ROI----------------------
    /*
    uint32_t imageWidth, imageLength;
    double**  LabeledImage= readLabeledImage(Im.ROIPath,&imageWidth, &imageLength);
    int  PixelDistance=3;

    vector<int> NeighborIDs;

    //The coordinates of convexHull Points
    for (int i = 0; i < contours.size(); i++) {
        Point sampleBorderPoint = contours[i][0];
        int ROILabel =LabeledImage[sampleBorderPoint.y+Im.ROIHeightBeg][sampleBorderPoint.x+Im.ROIWidthBeg];

        for (int j = 0; j < contours[i].size(); j++) {
            Point borderPoint = contours[i][j];
            int Px=borderPoint.x+Im.ROIWidthBeg;
            int Py=borderPoint.y+Im.ROIHeightBeg;

            for (int l=Py-PixelDistance; l=Py+PixelDistance; ++l)
                for (int k=Px-PixelDistance; k=Px+PixelDistance; ++k){

                    if (k<0) k=0;
                    if (k>imageWidth) k=imageWidth;
                    if (l<0) l=0;
                    if (l>imageLength) l=imageLength;

                    int value= LabeledImage[l][k];

                    if (value!=0 && value!=ROILabel)  NeighborIDs.push_back(value);
                }
        }
    }

        sort(NeighborIDs.begin(),NeighborIDs.end());
        NeighborIDs.erase(unique(NeighborIDs.begin(), NeighborIDs.end()),NeighborIDs.end());

        NeighborIDs.erase(std::find(NeighborIDs.begin(),NeighborIDs.end(),0));

        printf("Number of Neighboring ROIs is %d\n",NeighborIDs.size());

    delete [] LabeledImage;

*/

    //--------------------------------Hexagonality/Polygonality--------------------------------
    //This section is a translation from the following Python code
    //https://github.com/LabShare/polus-plugins/blob/master/polus-feature-extraction-plugin/src/main.py

    int neighbors=4; //NeighborIDs.size()
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
            sqrdTmp= (area_array[i]- area_ratio_ave)*(area_array[i]- area_ratio_ave);
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
            sqrdTmp2= (perim_array[i]- perim_ratio_ave)*(perim_array[i]- perim_ratio_ave);
        }
        double perim_ratio_sd=sqrt(sqrdTmp2/perim_array.size());

        //Set the HSR equal to the average Perimeter Ratio
        double hex_size_ratio = perim_ratio_ave;
        double hex_sd=sqrt((area_ratio_sd*area_ratio_sd+perim_ratio_sd*perim_ratio_sd)/2);
        double hex_ave = 10*(hex_area_ratio + hex_size_ratio)/2;

        ratios[1]=poly_size_ratio;
        ratios[2]=poly_area_ratio;
        ratios[3]=poly_ave;
        ratios[4]=hex_size_ratio;
        ratios[5]=hex_area_ratio;
        ratios[6]=hex_ave;
        ratios[7]=hex_sd;

    }
    else if (neighbors <3 ){
        double poly_size_ratio = std::numeric_limits<double>::quiet_NaN();
        double poly_area_ratio = std::numeric_limits<double>::quiet_NaN();
        double poly_ave = std::numeric_limits<double>::quiet_NaN();
        double hex_size_ratio = std::numeric_limits<double>::quiet_NaN();
        double hex_area_ratio = std::numeric_limits<double>::quiet_NaN();
        double hex_ave = std::numeric_limits<double>::quiet_NaN();
        double hex_sd = std::numeric_limits<double>::quiet_NaN();

        ratios[40]=poly_size_ratio;
        ratios[41]=poly_area_ratio;
        ratios[42]=poly_ave;
        ratios[43]=hex_size_ratio;
        ratios[44]=hex_area_ratio;
        ratios[45]=hex_ave;
        ratios[46]=hex_sd;
    }

    //------------------------------------------------------------------

    //Reconstruct LabelledImage

    unsigned char ** LabeledImageMatrix2 = new unsigned char *[Im.height];
    for (int i=0; i<Im.height; ++i) LabeledImageMatrix2[i] = new unsigned char [Im.width];


    //??  copyFields (matrix_IN);
    //??  allocate (matrix_IN.width, matrix_IN.height);

    //writeablePixels Labeled_plane = LabeledImageMatrix2.WriteablePixels();
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
            //                if (std::isnan(in_plane (y,x))) LabeledImageMatrix2[y][x]=std::numeric_limits<double>::quiet_NaN();
            if (std::isnan(in_plane (y,x))) LabeledImageMatrix2[y][x]=(unsigned char)0;
            else LabeledImageMatrix2[y][x]=(unsigned char)1;
        }

    long Euler= EulerNumber(LabeledImageMatrix2,8,Im.height,Im.width);

    delete [] LabeledImageMatrix2;

    //--------------------Mode and Entropy------------------------------------------
    int intMax=(int)max;
    int intMin=(int)min;
    int Size=intMax-intMin+1;

    int* histBins =new int [Size];

    for (int i=0; i<Size; ++i) histBins[i]=0;

    for (unsigned int y = 0; y < Im.height; ++y)
        for (unsigned int x = 0; x < Im.width; ++x){
            if (!std::isnan(in_plane (y,x))) {
                int bin = floor(in_plane (y,x));
                ++ histBins[bin-intMin];
            }
        }

    float MaxValue=0;
    int maxBinIndex=-1;
    double entropy = 0.0;
    for (int i=0; i<Size; i++) {
        float binEntry = (float)histBins[i]/Im.stats.n();
        if (fabs(binEntry) > 1e-6){  //if bin is not empty
            entropy -= binEntry * log2(binEntry);
            if (binEntry>MaxValue) {MaxValue=binEntry; maxBinIndex=i;}
        }
    }

    int ModeValue=maxBinIndex+intMin; //mode value

    cout << "Finished!\n" << endl;




}










