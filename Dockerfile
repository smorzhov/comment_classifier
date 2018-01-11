FROM gw000/keras:2.1.1-py3-tf-gpu

# Run the copied file and install some dependencies
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
    bash \
    python3-matplotlib && \
    apt-get clean && \
    pip3 --no-cache-dir install --upgrade \
    numpy \
    scipy \
    pandas \
    nltk \
    gensim

WORKDIR /comment_classifier

VOLUME ["/comment_classifier"]
RUN chmod +x /comment_classifier

ENTRYPOINT python3 install.py
