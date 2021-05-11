/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* Written by:                                                                   */
/*      Mahdi Maghrebi <mahdi.maghrebi [at] nih [dot] gov>                                */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#ifndef __MORPHOLOGICALALGORITHMS_H_
#define __MORPHOLOGICALALGORITHMS_H_

#include "cmatrix.h"
#include <vector>  

void GlobalCentroid2(const ImageMatrix &Im, double * x_centroid, double * y_centroid);

void MorphologicalAlgorithms(const ImageMatrix &Im, double *ratios);

long EulerNumber(unsigned char * pix_plane, int mode,int height, int width);

void Extrema (const ImageMatrix& Im, double *ratios);

struct Statistics{
    int min, max, mode;
    double mean, median, stdev;
};

Statistics ComputeCommonStatistics (std::vector<int> Data);

Statistics ComputeCommonStatistics2 (std::vector<double> Data);

#endif

