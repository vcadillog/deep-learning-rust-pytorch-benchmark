FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# set bash as current shell
# RUN chsh -s /bin/bash
# SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install -y wget zip curl git
RUN curl --proto '=https' https://sh.rustup.rs | sh -s -- -y
ENV PATH="$PATH:$HOME/.cargo/env"
RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip 
RUN unzip libtorch.zip && rm libtorch.zip 

ENV LIBTORCH=/libtorch
ENV LIBTORCH_INCLUDE=$LIBTORCH/include
ENV LIBTORCH_LIB=$LIBTORCH/lib
ENV LIBRARY_PATH="$LIBRARY_PATH:$LIBTORCH/lib"
ENV CPATH="$CPATH:$LIBTORCH_LIB:$LIBTORCH_INCLUDE:$LIBTORCH_INCLUDE/torch/csrc/api/include"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBTORCH_LIB"

RUN git clone https://github.com/vcadillog/deep-learning-rust-pytorch-benchmark.git

WORKDIR /deep-learning-rust-pytorch-benchmark/rust

# WORKDIR /workspace/
# COPY ./config.yaml .
# COPY ./data ./data
# COPY ./logs ./logs
# COPY ./rust ./rust

# CMD cd rust; cargo run --release

