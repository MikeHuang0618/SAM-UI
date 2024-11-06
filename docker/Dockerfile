FROM ubuntu:latest

LABEL maintainer "Mike <main668888@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

RUN \
  apt-get update -qq && \
  apt-get install -qq -y \
  git \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

WORKDIR /app

COPY ../requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY .. .
