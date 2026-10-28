# syntax=docker/dockerfile:1
# COLMAP 4.2.0 (CUDA, sm_120) linked against Ceres master (2.3) built with CUDA + cuDSS, so the GLOMAP
# global_mapper runs global positioning and bundle adjustment on the GPU (the conda COLMAP has a Ceres
# without cuDSS and falls back to the CPU sparse solvers). Used by src/sfm/container.py (sfm.container).
#
# Build (no context needed, sources are fetched at pinned versions):
#   docker build -t t2/colmap-cudss:4.2.0 docker/colmap-cudss
# Check:
#   docker run --rm --gpus all t2/colmap-cudss:4.2.0 colmap -h | head -2     # "... with CUDA"
#   docker run --rm t2/colmap-cudss:4.2.0 sh -c 'ldd $(which colmap) | grep cudss'

ARG CUDA_VERSION=12.8.1
ARG UBUNTU_VERSION=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS builder

ARG CERES_COMMIT=07a952622c2118433c783b1c3bd3fabe68420fc5
ARG COLMAP_VERSION=4.2.0
ARG CUDA_ARCHITECTURES=120
ARG JOBS=16
ENV DEBIAN_FRONTEND=noninteractive

# cuDSS 0.7.x comes from the NVIDIA CUDA apt repository already configured in the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates cmake ninja-build build-essential \
        libeigen3-dev libsuitesparse-dev libmetis-dev libopenblas-dev liblapack-dev \
        libboost-program-options-dev libboost-graph-dev libboost-system-dev \
        libgoogle-glog-dev libgflags-dev libsqlite3-dev libopenimageio-dev openimageio-tools \
        libcurl4-openssl-dev libssl-dev libgl-dev libglew-dev \
        'libcudss0-dev-cuda-12=0.7.*' 'libcudss0-cuda-12=0.7.*' \
    && rm -rf /var/lib/apt/lists/*

# The Ubuntu cuDSS package keeps cudss.h in /usr/include/libcudss/12, which its cudss-config.cmake does not
# search (it looks next to lib/x86_64-linux-gnu/libcudss/): link it there; drop the static-target file
# (libcudss_static.a ships in another package, Ceres links the shared lib); put the libs on the loader path.
RUN ln -s /usr/include/libcudss/12 /usr/lib/x86_64-linux-gnu/libcudss/include \
    && rm /usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss/cudss-static-targets*.cmake \
    && test -e /usr/lib/x86_64-linux-gnu/libcudss/12/libcudss.so \
    && echo /usr/lib/x86_64-linux-gnu/libcudss/12 > /etc/ld.so.conf.d/cudss.conf && ldconfig

# Ceres master with its abseil submodule (system abseil 20220623 is older than the required 20240116).
# Ceres overrides CMAKE_CUDA_ARCHITECTURES with a list ending at sm_90: add sm_120 (Blackwell).
RUN git clone https://github.com/ceres-solver/ceres-solver.git /src/ceres \
    && cd /src/ceres && git checkout ${CERES_COMMIT} && git submodule update --init --recursive \
    && sed -i 's/list(APPEND CMAKE_CUDA_ARCHITECTURES "90")/list(APPEND CMAKE_CUDA_ARCHITECTURES "90" "'${CUDA_ARCHITECTURES}'")/' CMakeLists.txt \
    && grep -q '"90" "'${CUDA_ARCHITECTURES}'"' CMakeLists.txt

RUN cudss_dir=$(dirname "$(find /usr -name cudss-config.cmake | head -1)") \
    && echo "cudss_DIR=${cudss_dir}" \
    && cmake -S /src/ceres -B /src/ceres/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_SHARED_LIBS=ON \
        -DWITH_CUDA=default -DWITH_SUITESPARSE=ON -Dcudss_DIR=${cudss_dir} \
        2>&1 | tee /src/ceres_cmake.log \
    && grep -E 'cuDSS|CUDA' /src/ceres_cmake.log \
    && ninja -C /src/ceres/build -j${JOBS} install

RUN git clone --depth 1 --branch ${COLMAP_VERSION} https://github.com/colmap/colmap.git /src/colmap \
    && mkdir -p /usr/include/opencv4 \
    && cmake -S /src/colmap -B /src/colmap/build -GNinja \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/colmap-install \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} -DCUDA_ENABLED=ON \
        -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF -DCGAL_ENABLED=OFF -DONNX_ENABLED=OFF -DTESTS_ENABLED=OFF \
        -DCCACHE_ENABLED=OFF -DBLA_VENDOR=OpenBLAS -Dcudss_DIR=/usr/lib/x86_64-linux-gnu/libcudss/12/cmake/cudss \
        2>&1 | tee /src/colmap_cmake.log \
    && ninja -C /src/colmap/build -j${JOBS} install

# Runtime: CUDA runtime libraries (cublas, cusolver, cusparse) + cuDSS + the shared libs COLMAP links.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-program-options1.83.0 libboost-graph1.83.0 libgomp1 libmetis5 \
        libgoogle-glog0v6t64 libgflags2.2 libsqlite3-0 libopenimageio2.4t64 libcurl4t64 libssl3t64 \
        libopenblas0 liblapack3 libcholmod5 libspqr4 libamd3 libcamd3 libccolamd3 libcolamd3 \
        libsuitesparseconfig7 libgl1 libopengl0 libglew2.2 \
        'libcudss0-cuda-12=0.7.*' \
    && rm -rf /var/lib/apt/lists/* \
    && echo /usr/lib/x86_64-linux-gnu/libcudss/12 > /etc/ld.so.conf.d/cudss.conf

COPY --from=builder /usr/local/lib/ /usr/local/lib/
COPY --from=builder /colmap-install/ /usr/local/
COPY --from=builder /src/ceres_cmake.log /src/colmap_cmake.log /usr/local/share/colmap-cudss/
RUN ldconfig && ! ldd /usr/local/bin/colmap | grep 'not found'

CMD ["colmap", "-h"]
