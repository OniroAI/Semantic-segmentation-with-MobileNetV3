FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

RUN apt-get update &&\
    apt-get -y install unzip git wget sysstat\
    python3 python3-pip python3-dev python3-setuptools\
    libsm6 libxext6 libxrender-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libprotoc-dev
RUN pip install onnx

RUN pip install tensorflow-gpu==2.0.0b1

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install pycocotools

ENV PYTHONPATH $PYTHONPATH:/workdir
ENV TENSORFLOW_HOME=/workdir/data/.tensorflow

WORKDIR /workdir
