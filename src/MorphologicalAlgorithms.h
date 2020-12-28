/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
/* Written by:                                                                   */
/*      Mahdi Maghrebi <mahdi.maghrebi [at] nih [dot] gov>                                */
/*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
#ifndef __MORPHOLOGICALALGORITHMS_H_
#define __MORPHOLOGICALALGORITHMS_H_

#include "cmatrix.h"

void GlobalCentroid2(const ImageMatrix &Im, double *x_centroid, double *y_centroid);

void MorphologicalAlgorithms(const ImageMatrix &Im, double *ratios);

#endif

