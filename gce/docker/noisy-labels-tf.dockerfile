# Ubuntu 18.04
# TensorFlow 1.13.0 CPU

FROM ubuntu:18.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# Environment
ENV CFLAGS="-O3"
ENV PYENV_ROOT=/root/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
ENV DATA_PATH=/data

# Linux packages
RUN apt-get update --fix-missing
RUN apt-get install -y \
    make cmake build-essential autoconf libtool rsync \
    ca-certificates git grep sed dpkg curl wget bzip2 unzip llvm \
    libssl-dev libreadline-dev libncurses5-dev libncursesw5-dev \
    libbz2-dev libsqlite3-dev zlib1g-dev \
    mpich htop vim tmux

# Setup tmux
RUN git clone https://github.com/gpakosz/.tmux.git /root/.tmux && \
    ln -s -f /root/.tmux/.tmux.conf /root/.tmux.conf && \
    cp /root/.tmux/.tmux.conf.local /root/

# Pyenv, Python 3, and Python packages
RUN git clone https://github.com/pyenv/pyenv.git $PYENV_ROOT && \
    pyenv install 3.6.6 && pyenv global 3.6.6 && pyenv rehash
RUN pip install -U \
    pip \
    ipython \
    numpy scipy \
    cloudpickle \
    scikit-image \
    requests \
    click \
    xlrd \
    pandas \
    pyyaml \
    tqdm \
    bert-serving-client \
    tensorflow==1.13.1

# Add more stuff as needed below
COPY run-experiment.sh /

