# Toxic comment classifier

## Description

## Prerequisites

You will need the following things properly installed on your computer.

* [Docker](https://www.docker.com/)

## Installation

* `git clone https://github.com/smorzhov/comment_classifier`

## Running

Remember that Docker container has the Python version 2.7.12!

1. Download and unrar (unzip) [test](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/download/test.csv.zip) and [train](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/download/train.csv.zip) data into `~src/data` directory.
1. Download pretrained [word2vec](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit) Gooogle model, [glove_6B](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) model and [globe_840B](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and unpack them into `~src/data/raw` directory.
2. If you are planning to use nvidia-docker, you need to building nvidia-docker image first. Otherwise, you can skip this step
    ```bash
    nvidia-docker build -t sm_keras_tf:gpu .
    ```
    Run container
    ```bash
    nvidia-docker run -v $PWD/src:/comment_classifier -dt --name tcc sm_keras_tf:gpu /bin/bash
    ```
3. Training
    ```bash
    nvidia-docker exec tcc python train.py [-h]
    ```

## Advices

You can add some custom stop words. They must be placed in `~src/data/stopwords.txt` file (one word per line).

You can create some files with useful information about training data
```bash
nvidia-docker exec tcc python info.py
```