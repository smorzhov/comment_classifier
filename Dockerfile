FROM gw000/keras:2.1.1-py3-tf-gpu

COPY install.py install.py

VOLUME ["/comment_classifier"]

# Run the copied file and install some dependencies
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
    python3-matplotlib && \
    apt-get clean && \
    chmod +x /comment_classifier && \
    pip3 --no-cache-dir install --upgrade \
    gensim \
    nltk \
    numpy \
    pandas \
    scipy && \
    python3 install.py && \
    rm -f install.py

WORKDIR /comment_classifier