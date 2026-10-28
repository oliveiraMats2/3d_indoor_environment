# OpenMVS (InterfaceCOLMAP, DensifyPointCloud, ...) with CUDA PatchMatch for an RTX 50xx (sm_120).
# Follows upstream docker/buildInDocker.sh (Eigen 3.4, CGAL 6.0.1, VCG, apt Boost; OpenCV 4.12.0) with every
# source pinned to a commit. Used by src/dense/openmvs.py through dense.openmvs.container.
#
#   docker build -t t2/openmvs:2.4.0 docker/openmvs
#   docker run --rm --gpus all t2/openmvs:2.4.0 DensifyPointCloud --help
#
# The container runs as the caller (--user uid:gid), so no user is created here.

ARG BASE_IMAGE=nvidia/cuda:12.8.1-devel-ubuntu24.04
FROM ${BASE_IMAGE}

ARG OPENMVS_TAG=v2.4.0
ARG OPENMVS_SHA=58117204c86bbb11a0b25b26a8987676cf11274d
ARG EIGEN_SHA=1ac855c25dc4aa8453c8e18e0db55336b6bdd0ba
ARG CGAL_SHA=2affe725b47e3f408e398eec339a1ac82fd64561
ARG OPENCV_SHA=cbee6841638edb6fbc8110df7cd52bb8e3d66211
ARG VCG_SHA=658ba36d0a5666650da6e066b4794efc5a463407
ARG CUDA_ARCHITECTURES=120
ARG JOBS=16

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update -yq && apt-get install -yq --no-install-recommends \
        build-essential git cmake ca-certificates \
        libpng-dev libjpeg-dev libtiff-dev libglu1-mesa-dev libglew-dev libglfw3-dev \
        libboost-iostreams-dev libboost-program-options-dev libboost-system-dev libboost-serialization-dev \
        libboost-thread-dev libgmp-dev libmpfr-dev zlib1g-dev python3-dev \
        libnanoflann-dev libjxl-dev libwebp-dev libopenexr-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# fetch <url> <sha> <dir>: shallow checkout of one pinned commit
RUN printf '#!/bin/bash\nset -e\ngit init -q "$3" && git -C "$3" fetch -q --depth 1 "$1" "$2" && git -C "$3" checkout -q FETCH_HEAD\n' \
        > /usr/local/bin/fetch && chmod +x /usr/local/bin/fetch

WORKDIR /build

RUN fetch https://gitlab.com/libeigen/eigen.git ${EIGEN_SHA} eigen \
    && cmake -S eigen -B eigen_build -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
    && cmake --install eigen_build && rm -rf eigen eigen_build

RUN fetch https://github.com/cgal/cgal.git ${CGAL_SHA} cgal \
    && cmake -S cgal -B cgal_build -DCMAKE_BUILD_TYPE=Release \
    && cmake --install cgal_build && rm -rf cgal cgal_build

# OpenMVS 2.4 writes JPEG-XL through cv::IMWRITE_JPEGXL_QUALITY (OpenCV >= 4.11); Ubuntu 24.04 ships 4.6,
# so OpenCV is built from source with the modules OpenMVS uses (no OpenCV CUDA: OpenMVS has its own kernels)
RUN fetch https://github.com/opencv/opencv.git ${OPENCV_SHA} opencv \
    && cmake -S opencv -B opencv_build -DCMAKE_BUILD_TYPE=Release -DBUILD_LIST=core,imgproc,imgcodecs,calib3d,features2d,flann,highgui \
        -DWITH_JPEGXL=ON -DWITH_CUDA=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
        -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python3=OFF \
    && cmake --build opencv_build -j ${JOBS} && cmake --install opencv_build && ldconfig \
    && rm -rf opencv opencv_build

RUN fetch https://github.com/cdcseacave/VCG.git ${VCG_SHA} /vcglib \
    && fetch https://github.com/cdcseacave/openMVS.git ${OPENMVS_SHA} openMVS \
    && cmake -S openMVS -B openMVS_build -DCMAKE_BUILD_TYPE=Release -DVCG_ROOT=/vcglib -DOpenMVS_BUILD_VIEWER=OFF \
        -DOpenMVS_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
        -DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/ -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda/ \
        -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3 \
    && cmake --build openMVS_build -j ${JOBS} \
    && cmake --install openMVS_build \
    && rm -rf openMVS openMVS_build /vcglib

ENV PATH=/usr/local/bin/OpenMVS:${PATH}
LABEL org.opencontainers.image.source="https://github.com/cdcseacave/openMVS" \
      org.opencontainers.image.version="${OPENMVS_TAG}"
WORKDIR /work
