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

COPY . /home/
WORKDIR /home/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/fftw/lib:/usr/lib/x86_64-linux-gnu
ENV LDFLAGS="-L/home/fftw/lib -L/usr/lib/x86_64-linux-gnu"
ENV CPPFLAGS="-I/home/fftw/include"

RUN ./configure --prefix=/home/ LIBS='-L/usr/lib/x86_64-linux-gnu/ -lboost_iostreams  -lboost_system -lboost_filesystem' CXXFLAGS='-g -O2 -fopenmp'

RUN make install

RUN mkdir /home/input 
RUN mkdir /home/output 
RUN mkdir /home/mask 

ENTRYPOINT ["/home/bin/wndchrm"]

#sudo docker build -t labshare/polus-wnd-charm-plugin:0.2.0 .

#sudo docker run -v /home/maghrebim2/Work/WND-CHARM/ROI/MultiThreading/Working_Dir2/Data:/home/input -v /home/maghrebim2/Work/WND-CHARM/ROI/MultiThreading/Working_Dir2/LabeledData:/home/mask -v /home/maghrebim2/Work/WND-CHARM/ROI/MultiThreading/Working_Dir2/Output:/home/output  labshare/polus-wnd-charm-plugin:0.2.0 --DataPath /home/input --LabeledData /home/mask  --ImageTransformationName  Chebyshev-Fourier_1D_ColumnWise  --FeatureAlgorithmName PixelStatistics --output /home/output 
