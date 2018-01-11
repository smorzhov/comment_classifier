# Toxic comment classifier

## Description

## Prerequisites

You will need the following things properly installed on your computer.

* [Docker](https://www.docker.com/)

## Installation

* `git clone https://github.com/smorzhov/comment_classifier`

## Running

1. Download and unrar [test](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/download/test.csv.zip) and [train](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/download/train.csv.zip) data into `~src/data` directory. 
2. If you are planning to use nvidia-docker, you need to building nvidia-docker image first. Otherwise, you can skip this step
    ```bash
    nvidia-docker build -t sm_keras:gpu .
    ```
    Run container
    ```bash
    nvidia-docker run -v $PWD/src:/comment_classifier -dt --name tcc sm_keras:gpu /bin/bash
    ```
3. Create some files with useful information about training data
    ```bash
    nvidia-docker exec tcc python info.py
    ```