FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get -y update && \
    apt-get -y install gcc wget zip && \
    apt-get -y  install python3-gdbm && \
    apt-get -y install g++ make
 
RUN mkdir -p /home/fftw /home/tiff
WORKDIR /home/fftw
RUN wget http://www.fftw.org/fftw-3.3.8.tar.gz
RUN tar xfz fftw-3.3.8.tar.gz
RUN rm fftw-3.3.8.tar.gz
WORKDIR /home/fftw/fftw-3.3.8/
RUN ./configure --prefix=/home/fftw
RUN make install

WORKDIR /home/tiff 
RUN wget http://download.osgeo.org/libtiff/tiff-4.1.0.zip 
RUN unzip tiff-4.1.0.zip
RUN rm tiff-4.1.0.zip
WORKDIR /home/tiff/tiff-4.1.0
RUN ./configure --prefix=/home/tiff
RUN make install

COPY . /home/
WORKDIR /home
ENV LDFLAGS="-L/home/fftw/lib -L/home/tiff/lib"
ENV CPPFLAGS="-I/home/fftw/include -I/home/tiff/include"

ENV LIBRARY_PATH="/home/fftw/lib:${LIBRARY_PATH}" 
ENV LIBRARY_PATH="/home/tiff/lib:${LIBRARY_PATH}" 

RUN ./configure --prefix=/home/
RUN make install

ENTRYPOINT ["/home/bin/wndchrm"]
