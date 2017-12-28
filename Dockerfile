FROM  ???

# Copy the file into docker
COPY requirements.txt requirements.txt

# Run the copied file
RUN pip install -r requirements.txt && \
    pip install git+https://github.com/aleju/imgaug && \
    apt-get clean && \
    rm requirements.txt && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /comment_classifier

VOLUME ["/comment_classifier"]