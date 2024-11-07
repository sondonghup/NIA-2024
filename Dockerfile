FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

# 기본 설정
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

# Python 설치 및 업데이트
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# COPY . /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cu122
RUN pip install torchvision
