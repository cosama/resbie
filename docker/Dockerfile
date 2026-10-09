ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS builder

ENV DEBIAN_FRONTEND=noninteractive

# The empty libCppUnitLite.a stub: libgtsam-dev's CMake exports reference
# GTSAM's unit-test library, which the package does not ship, so
# find_package(GTSAM) fails without it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        libeigen3-dev \
        libgtsam-dev \
        libmetis-dev \
        libpcl-dev \
        libsuitesparse-dev \
        libtbb-dev \
        libyaml-cpp-dev \
        ninja-build \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && ar cr /usr/lib/x86_64-linux-gnu/libCppUnitLite.a

WORKDIR /src
COPY core /src/core

RUN python3 -m venv /opt/build-venv \
    && /opt/build-venv/bin/pip install --no-cache-dir --upgrade pip wheel \
    && /opt/build-venv/bin/pip wheel --no-cache-dir --wheel-dir /wheelhouse \
        ./core "rosbags>=0.11" "ruamel.yaml"

FROM ubuntu:${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/resbie/bin:${PATH}
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libgtsam4 \
        libmetis5 \
        libpcl-common1.14 \
        libpcl-filters1.14 \
        libpcl-io1.14 \
        libpcl-registration1.14 \
        libtbb12 \
        libyaml-cpp0.8 \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheelhouse /wheelhouse

RUN python3 -m venv /opt/resbie \
    && /opt/resbie/bin/pip install --no-cache-dir --no-index --find-links=/wheelhouse \
        resbie-offline rosbags ruamel.yaml \
    && rm -rf /wheelhouse

WORKDIR /data

CMD ["python", "-c", "import resbie, rosbags; from ruamel.yaml import YAML; print('resbie-offline ready')"]
