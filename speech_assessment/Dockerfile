# syntax=docker/dockerfile:1.4
FROM --platform=$BUILDPLATFORM python:3.12.3 AS builder
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg
RUN mkdir /models
WORKDIR /models
COPY requirements.txt /models/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt 
COPY . .