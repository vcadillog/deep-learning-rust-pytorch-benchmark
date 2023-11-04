FROM rust:1.73

RUN apt-get install unzip

# RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
RUN wget -O libtorch.zip https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip 

RUN unzip libtorch.zip && rm libtorch.zip 

ENV LIBTORCH=/libtorch
ENV LIBTORCH_INCLUDE=$LIBTORCH/include
ENV LIBTORCH_LIB=$LIBTORCH/lib
ENV LIBRARY_PATH="$LIBRARY_PATH:$LIBTORCH/lib"
ENV CPATH="$CPATH:$LIBTORCH_LIB:$LIBTORCH_INCLUDE:$LIBTORCH_INCLUDE/torch/csrc/api/include"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$LIBTORCH_LIB"


WORKDIR /workspaces/
COPY ./config.yaml .
COPY ./data ./data
COPY ./logs ./logs
COPY ./rust ./rust

CMD cd rust; cargo run --release

