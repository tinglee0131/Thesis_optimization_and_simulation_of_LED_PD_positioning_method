FROM python:3.12-slim

WORKDIR /app

# install system dependency
# # better to run in one line, multiple lines create bigger image
# RUN apt-get update
# # buile-essential: debian-based distribution(ubuntu) packages
# # include gccm make etc, for compiling c(used in numpy and so)
# RUN apt-get install -y --no-install-recommends build-essential
# # clear up the list for space efficiency
# RUN rm -rf /var/lib/apt/lists/* 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tk-dev \
    python3-tk \
    libx11-dev \
    libxft-dev \
    libxss-dev \
    libxext-dev \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y fonts-noto
# install python dependency
COPY requirements.txt .
# do not store download package files
# files downloaded are installed and put necessary files into the system, the remaining downloaded file is stored for future installation cache only, not for running
RUN pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# set environment variables
ENV PYTHONPATH=/app

# Set entrypoint
# represent in exec form(json)
# ENTRYPOINT ["python", "main.py"]
ENTRYPOINT ["/bin/bash"]

# Default command
# with entrypoint: setting default variables
# without entrypoint, cmd will be the entrypoint
# only the last CMD has effect
# CMD ["--mode", "scenario"]