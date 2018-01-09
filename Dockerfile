FROM  ???

# Copy the file into docker
COPY requirements.txt requirements.txt

# Run the copied file
RUN pip install -r requirements.txt && \
    apt-get clean && \
    rm requirements.txt && \
    rm -rf /var/lib/apt/lists/*
    

WORKDIR /comment_classifier

VOLUME ["/comment_classifier"]

ENTRYPOINT python install.py
