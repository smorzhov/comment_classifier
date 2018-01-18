FROM nvcr.io/nvidia/tensorflow:17.10

COPY install.py install.py

VOLUME ["/comment_classifier"]

# Run the copied file and install some dependencies
RUN apt update -qq && \
    apt install --no-install-recommends -y \
    # install essentials
    build-essential \
    g++ \
    git \
    graphviz \
    openssh-client \
    # requirements for numpy
    libopenblas-base \
    python-numpy \
    python-scipy \
    # requirements for keras
    python-h5py \
    python-yaml \
    python-pydot \
    # requirements for pydot
    python-pydot \
    python-pydot-ng && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    chmod +x /comment_classifier && \
    pip --no-cache-dir install --upgrade \
    gensim \
    keras \
    matplotlib \
    nltk \
    pandas && \
    python install.py && \
    rm -f install.py

WORKDIR /comment_classifier