# WND-CHARM
WND-CHARM is a multi-purpose image classifier that can be applied to a wide variety of image classification tasks without modifications or fine-tuning, and yet provides classification accuracy comparable to state-of-the-art task-specific image classifiers. WND-CHARM can extract up to ~3,000 generic image descriptors (features) including polynomial decompositions, high contrast features, pixel statistics, and textures. These features are derived from the raw image, transforms of the image, and compound transforms of the image (transforms of transforms). The features are filtered and weighted depending on their effectiveness in discriminating between a set of predefined image classes (the training set). These features are then used to classify test images based on their similarity to the training classes. This classifier was tested on a wide variety of imaging problems including biological and medical image classification using several imaging modalities, face recognition, and other pattern recognition tasks. WND-CHARM is an acronym that stands for *"Weighted Neighbor Distance using Compound Hierarchy of Algorithms Representing Morphology."*

This package contains two implementations both of which use common image transform and feature extraction code:

* A command-line program `wndchrm` (without the a) written in C++ that streamlines the WND-CHARM algorithm workflow. It reads images and their class membership from a directory hierarchy or text file, and outputs classifier statistics to an HTML report or STDOUT. To build from distribution tarball, use `./configure && make`. To build from a cloned repository use `./build.sh`. 
* A Python library `wndcharm` that provides an API to do many of the same things as wndchrm while providing the flexibility of a scripting language to perform low manipulation and visualization of pixel intensities, generated features and classification results. To build, use `python setup.py build`.

This research was supported entirely by the Intramural Research Program of the National Institutes of Health, National Institute on Aging, Ilya Goldberg, Investigator. Address: Laboratory of Genetics/NIA/NIH, 251 Bayview Blvd., Suite 100, Baltimore, MD, 21224, USA

----
#### A full description of the wndchrm utility can be found at:

[Shamir L, Orlov N, Eckley DM, Macura T, Johnston J, Goldberg IG. Wndchrm - an open source utility for biological image analysis](http://www.scfbm.org/content/3/1/13). BMC Source Code for Biology and Medicine. 3: 13, 2008. [PDF download](https://ome.irp.nia.nih.gov/wnd-charm/BMC-wndchrm-utility.pdf)

#### The wndchrm utility is an implementation of the WND-CHARM algorithm described here:

[Orlov N, Shamir L, Macura T, Johnston J, Eckley DM, Goldberg IG. WND-CHARM: Multi-purpose image classification using compound image transforms](https://ome.irp.nia.nih.gov/wnd-charm/PRL_2008.pdf). Pattern Recognition Letters. 29(11): 1684-93, 2008.

#### A review of techniques used in pattern-recognition/machine learning as applied to image analysis is here:

[Shamir L, Delaney JD, Orlov N, Eckley DM, Goldberg IG. Pattern Recognition Software and Techniques for Biological Image Analysis](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000974). PLoS Computational Biology. 6(11): e1000974, 2010

## Current Release

The current release of the wndchrm command-line utility is  version 1.60, available here: [wndchrm-1.60.977f5f6.tar.gz](https://github.com/wnd-charm/wnd-charm/files/565545/wndchrm-1.60.977f5f6.tar.gz)

The previous release, v1.52, is available here: [wndchrm-1.52.775.tar.gz](https://github.com/wnd-charm/wnd-charm/files/565579/wndchrm-1.52.775.tar.gz)

## Supported Platforms

WND-CHARM should compile and run on any POSIX-compliant operating system. It has been tested on Linux (Ubuntu 12.04 w/ GCC 4.6, CentOS 6.3 w/ GCC 4.4) and Mac OS X (<=10.7 w/ GCC 4.2, with experimental support for 10.9 Mavericks w/ `clang` compiler).

## Dependencies

Installation of WND-CHARM minimally requires a C++ compiler, LibTIFF and FFTW.

* C++ Compiler
    * Mac OS X: Install the command-line developer tools.
    * Ubuntu/Debian: `sudo apt-get install build-essential`
* [LibTIFF 3.x](http://www.libtiff.org):
    * CentOS/RedHat: `sudo yum install libtiff-devel`
    * Ubuntu/Debian: `sudo apt-get install libtiff4-dev`
    * Mac OS X: `brew install libtiff`
* [FFTW 3.x](http://www.fftw.org/download.html):
    * CentOS/RedHat: `sudo yum install fftw-static fftw-devel`
    * Ubuntu/Debian: `sudo apt-get install libfftw3-dev`
    * Mac OS X: `brew install fftw`
* Optional for dendrograms: [PHYLIP](http://evolution.genetics.washington.edu/phylip/install.html)
    * Some X11 libraries must be installed prior to compiling/installing PHYLIP (`make install` in the src dir.)
        * CentOS/RedHat: `sudo yum install libX11-devel libXt-devel libXaw-devel`
        * Ubuntu/Debian: `sudo apt-get install libX11-dev libxt-dev libxaw7-dev`

#### WND-CHARM Python API additional dependencies
The WND-CHARM Python API additionally requires the Python development package, SWIG, and the common Python 3rd-party packages `numpy` and `scipy`. Optionally, result visualization tools are enabled by installing the package `matplotlib`. To run the provided example scripts, the package `argparse` is required (included with Python 2.7+).

* Python utilities:
    * CentOS/RedHat: `sudo yum install python-devel swig`
    * Ubuntu/Debian: `sudo apt-get install python-dev swig`
    * Mac OS X: `brew install python swig`
* Python packages:
    * CentOS/RedHat: `sudo yum install numpy scipy python-matplotlib argparse`
    * Ubuntu/Debian: `sudo apt-get install python-numpy python-scipy python-matplotlib`
    * Pip: `pip install numpy scipy matplotlib argparse`

## Recommended Hardware

2 GB RAM (per core), 10 GB HD space, 2 GHZ CPU. Please be aware that this utility is very computationally intensive. Multiple instances of wndchrm can work concurrently on the same dataset on multi-core/multi-processor CPUs. Simply use the same command line to launch as many instances of wndchrm as there are CPU cores available.

## Test Images

Please also visit the [IICBU Biological Image Repository](https://ome.irp.nia.nih.gov/iicbu2008), which provides a benchmark for testing and comparing the performance of image analysis algorithms for biological imaging.

## What is new?

The performance of WND-CHARM was profiled and improved using the following techniques. Also, the functionality to read tiled-tiff images was added in src/cmatrix.cpp using libtiff library.

* Multi-threading capability (OpenMP directives) was added for the performance bottlenecks which were in the following codes. 
    * /src/statistics/FeatureStatistics.cpp
    * /src/textures/gabor.cpp
    * /src/transforms/chebyshev.cpp
* Optimization technique of Data Locality was implememted in the following code. 
    * /src/statistics/CombFirst4Moments.cpp 
    
The number of threads can be set using `export OMP_NUM_THREADS=10`. Overall, the performance of WND-CHARM was improved by 45% using the new changes and by running with 10 threads.

In addition to the above changes, unit tests for the current implementations in WND-CHARM were tested and successfully passed. More importantly, the unit tests for the original WND-CHARM were reviewed and fixed by modifying the reference values for Haralick Textures (components 16 and 17) in the following files. The unit tests are now operational for the original WND-CHARM as well as the current implementations.

/tests/pywndcharm_tests/lymphoma_eosin_channel_MCL_test_img_sj-05-3362-R2_001_E-t6x5_5_4-l.sig /tests/pywndcharm_tests/lymphoma_eosin_channel_MCL_test_img_sj-05-3362-R2_001_E_t6x5_REFERENCE_SIGFILES.zip /tests/wndchrm_tests/010067_301x300-l_precalculated.sig

In addition to the above changes, the state of the art functionality "Region Of Interest (ROI)" was implemented in WND-CHARM. If Mask (Labeled) Image is provided as an input argument, WND-CHARM automatically computes the features for ROIs (non-zeor labels) instead of the entire image. The ROI performance was significantly improved by confining the feature computations to a rectangular bounding box around ROI.
Furthermore, the performance of ROI implementation was improved by multi-threading at the ROI level where each thread picks one ROI and computes the entire features for it. The input parameters to WND-CHARM were modified according to the requirements of a WIPP plugin. The outputs were also formatted according to the desired formats for WIPP plugins. WND-CHARM can now be executive using the following command and sets of input arguments.

./wndchrm --DataPath /PATH/to/Input Intensity Image/Directory --output /PATH/to/Output/Directory --LabeledData /PATH/to/Mask Image/Directory --ImageTransformationName Original  --FeatureAlgorithmName PixelIntensityStatistics

In the above command:

* DataPath: refers to the directory which contains the input tiff intensity image.
* output:   refers to the directory where the computed features file (in .csv format instead of .sig) is saved.
* LabeledData: refers to the directory which contains the mask tiff image for ROI computation. This parameter is optional and WND-CHARM computes the features for the entire image if LabeledData is not specified. Please note that the pair of intensity and mask images inside DataPath and LabeledData directories should have the same filename. This is how the code matches the masks and intensity images when multiple of them are present inside the DataPath and LabeledData directories.
* ImageTransformationName: The (optional) specific image transformation algorithm which is desired for the computation. Please refer to Tasks.cpp to learn more about the names of the available algorithms. Please note that the algorithms with the either names Fourier_1D_ColumnWise, Fourier_1D_RowWise, and Fourier_2D are additional algorithms specific to ROI case. 

* FeatureAlgorithmName: The (optional) specific feature extraction algorithm which is desired for the computation over the chosen ImageTransformationName. Please refer to Tasks.cpp to learn more about the names of the available algorithms. 

If ImageTransformationName and FeatureAlgorithmName are not specified, WND-CHARM computes the features for a short list of FeatureAlgorithmName and ImageTransformationName. Also, the long set of features can be alternatively selected using the input argument "--DesiredFeatures LongSet".

In addition to the above changes, a comprehensive list of Morphological algorithms was implemented in WND-CHARM at src/MorphologicalAlgorithms.cpp and can be invoked as an input argument using "--FeatureAlgorithmName Morphological". Morphological Features output the computed values for a total of 54 parameters which are described in the order below. It should be noted that for the common parameters with MATLAB's regionprops module, the computed values were converted to produce the same results as MATLAB.

0-Total number of the ROI (Region of Interest) pixels
1-x coordinate of the ROI's centroid
2-y coordinate of the ROI's centroid
3-x coordinate where the rectangular bounding box encompassing the ROI begins
4-y coordinate where the rectangular bounding box encompassing the ROI begins
5-Width of the rectangular bounding box encompassing the ROI
6-Height of the rectangular bounding box encompassing the ROI
7-MajorAxisLength: the length of major axis of the ellipse that has the same normalized second central moments as the region.
8-MinorAxisLength: the length of minor axis of the ellipse that has the same normalized second central moments as the region.
9-Eccentricity: ratio of focal distance over the major axis length.
10-Orientation: angle between the x axis and the major axis of the ellipse that has same second moments as the region.
11-Area of the convex hull
12-Circularity: roundness of the ROI which is computed as (4*Area*pi)/(Perimeter2). For a perfect circle, the circularity value is 1.
13-Filled Area: the same as the first feature. It is repeated here solely for consistency with MATLAB
14-Euler number: Euler characteristic of the ROI
15-x coordinate of the Extrema point at top-left
16-x coordinate of the Extrema point at top-right
17-x coordinate of the Extrema point at right-top
18-x coordinate of the Extrema point at right-bottom
19-x coordinate of the Extrema point at bottom-right
20-x coordinate of the Extrema point at bottom-left
21-x coordinate of the Extrema point at left-bottom
22-x coordinate of the Extrema point at left-top
23-y coordinate of the Extrema point at top-left
24-y coordinate of the Extrema point at top-right
25-y coordinate of the Extrema point at right-top
26-y coordinate of the Extrema point at right-bottom
27-y coordinate of the Extrema point at bottom-right
28-y coordinate of the Extrema point at bottom-left
29-y coordinate of the Extrema point at left-bottom
30-y coordinate of the Extrema point at left-top
31-Equivalent diameter: the diameter of a circle with the same area as the ROI
32-Solidity: the ratio of the pixel counts in the ROI to the pixel counts in the convex hull
33-Extent: the ratio of the pixel counts in the ROI to the pixel counts in the bounding box encompassing the ROI 
34-Perimeter length of the ROI
35-x coordinate of the ROI's weighted centroid
36-y coordinate of the ROI's weighted centroid 
37-Mean of the ROI pixels
38-Min of the ROI pixels
39-Max of the ROI pixels
40-Max Feret diamater
41-Max Feret angle
42-Min Feret diamater
43-Min Feret angle
44-Neighbors: the number of neighbors touching the ROI
45-Polygonality score: the score ranges from -infinity to 10. Score 10 indicates the object shape is polygon and score -infinity indicates the object shape is not polygon.
46-Hexagonality score: the score ranges from -infinity to 10. Score 10 indicates the object shape is hexagon and score -infinity indicates the object shape is not hexagon.
47-Hexagonality standard deviation: dispersion of hexagonality score relative to its mean.
48-Kurtosis of the ROI pixels
49-Median of the ROI pixels
50-Mode of the ROI pixels
51-Standard deviation of the ROI pixels 
52-Skewness of the ROI pixels
53-Entropy: a measure of randomness. It is the amount of information in the ROI.
