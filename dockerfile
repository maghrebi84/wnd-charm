FROM ubuntu:18.04  

# Installing  packages for the build
RUN apt-get -y update  && apt-get install -y \
build-essential \
cmake \
libboost-all-dev \
libtiff5-dev \
wget \
make \
unzip

#Installing fftw
RUN mkdir /home/fftw
WORKDIR /home/fftw
RUN wget http://www.fftw.org/fftw-3.3.8.tar.gz
RUN tar xfz fftw-3.3.8.tar.gz
RUN rm fftw-3.3.8.tar.gz
WORKDIR /home/fftw/fftw-3.3.8/
RUN ./configure --prefix=/home/fftw
RUN make install

#Installing OpenCV
RUN mkdir /home/OpenCV
WORKDIR /home/OpenCV
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/master.zip
RUN unzip opencv.zip
# Create build directory
RUN mkdir -p build 
WORKDIR /home/OpenCV/build
# Configure
RUN cmake  ../opencv-master
# Build
RUN cmake --build .

COPY . /home/
WORKDIR /home/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/fftw/lib:/usr/lib/x86_64-linux-gnu:/home/OpenCV/build/lib
ENV LDFLAGS="-L/home/fftw/lib -L/usr/lib/x86_64-linux-gnu -L/home/OpenCV/build/lib"
ENV CPPFLAGS="-I/home/fftw/include -I/home/OpenCV/opencv-master/modules/highgui/include -I/home/OpenCV/opencv-master/modules/imgproc/include -I/home/OpenCV/opencv-master/modules/core/include -I/home/OpenCV/build -I/home/OpenCV/opencv-master/modules/imgcodecs/include -I/home/OpenCV/opencv-master/modules/videoio/include"

RUN ./configure --prefix=/home/ LIBS='-L/home/OpenCV/build/lib -lopencv_imgproc -lopencv_highgui -lopencv_core -L/usr/lib/x86_64-linux-gnu/ -lboost_iostreams -lboost_system -lboost_filesystem' CXXFLAGS='-g -O2 -fopenmp -I/home/OpenCV/opencv-master/modules/highgui/include -I/home/OpenCV/opencv-master/modules/imgproc/include -I/home/OpenCV/opencv-master/modules/core/include -I/home/OpenCV/build -I/home/OpenCV/opencv-master/modules/imgcodecs/include -I/home/OpenCV/opencv-master/modules/videoio/include'

RUN make install

RUN mkdir /home/input 
RUN mkdir /home/output 
RUN mkdir /home/mask 

ENTRYPOINT ["/home/bin/wndchrm"]


#sudo docker build -t labshare/polus-wnd-charm-plugin:0.2.0 .

#sudo docker run -v /home/maghrebim2/Work/WND-CHARM/ROI/Morphology-OpenCV/wnd-charm15/Data:/home/input -v /home/maghrebim2/Work/WND-CHARM/ROI/Morphology-OpenCV/wnd-charm15/LabeledData:/home/mask -v /home/maghrebim2/Work/WND-CHARM/ROI/Morphology-OpenCV/wnd-charm15/Output:/home/output labshare/polus-wnd-charm-plugin:0.2.0 --DataPath /home/input --LabeledData /home/mask  --ImageTransformationName  Original  --FeatureAlgorithmName Morphological --output /home/output 

