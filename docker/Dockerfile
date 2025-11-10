FROM rocker/r-ver:4.2.2
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update -qq && apt-get install -y \
    ca-certificates wget curl git \
    build-essential pkg-config \
    libcurl4-openssl-dev libssl-dev \
    libpcre2-dev liblzma-dev libbz2-dev libicu-dev libffi-dev libudunits2-dev \
    python3.10 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Apache Arrow repo + C++ libs
RUN wget -O /tmp/arrow.deb https://apache.jfrog.io/artifactory/arrow/ubuntu/apache-arrow-apt-source-latest-jammy.deb \
  && dpkg -i /tmp/arrow.deb && rm /tmp/arrow.deb \
  && apt-get update \
  && apt-get install -y --no-install-recommends \
          libarrow-dev \
          libarrow-dataset-dev \
          libparquet-dev \
  && rm -rf /var/lib/apt/lists/*

ENV PKG_CONFIG_PATH=/usr/lib/aarch64-linux-gnu/pkgconfig:/usr/lib/x86_64-linux-gnu/pkgconfig
ENV ARROW_USE_PKG_CONFIG=TRUE ARROW_R_DEV=FALSE

# Python userland
RUN ln -s /usr/bin/python3.10 /usr/local/bin/python
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install \
    "pyarrow==22.0.0" \
    "pandas==2.2.3" \
    "numpy==2.2.5" \
    "rpy2==3.5.12" \
    "pyyaml==6.0.3"

WORKDIR /home/ruser/app

