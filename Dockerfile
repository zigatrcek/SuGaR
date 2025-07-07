FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PATH="/opt/conda/bin:${PATH}"
# Set CUDA architecture flags for extension compilation
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6+PTX"
# Set necessary environment variables for OpenGL
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
ENV NVIDIA_VISIBLE_DEVICES=all

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    cmake \
    ninja-build \
    g++ \
    libglew-dev \
    libassimp-dev \
    libboost-all-dev \
    libgtk-3-dev \
    libopencv-dev \
    libglfw3-dev \
    libavdevice-dev \
    libavcodec-dev \
    libeigen3-dev \
    libxxf86vm-dev \
    libembree-dev \
    libtbb-dev \
    ca-certificates \
    ffmpeg \
    curl \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    # Enhanced OpenGL support packages
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libegl1-mesa \
    libegl1 \
    libgles2-mesa-dev \
    libglvnd0 \
    libglx0 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxxf86vm1 \
    libglu1-mesa \
    xvfb \
    mesa-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Ensure correct permissions for /dev/dri if available in the container
RUN mkdir -p /usr/share/glvnd/egl_vendor.d
RUN echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/conda \
    && rm miniconda.sh

# Set working directory
WORKDIR /app

# Copy the SuGaR repository files (instead of cloning from GitHub)
COPY . .

# Initialize conda for bash
RUN conda init bash

# Create the conda environment first
RUN conda env create -f environment.yml

# Build and install the CUDA extensions with proper conda environment
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate sugar && \
    cd /app/gaussian_splatting/submodules/diff-gaussian-rasterization && \
    pip install -e . && \
    cd ../simple-knn && \
    pip install -e .

# Install nvdiffrast
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate sugar && \
    pip install nvdiffrast

# Create symbolic links for the modules if needed
RUN ln -sf /app/gaussian_splatting/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization /app/gaussian_splatting/ && \
    ln -sf /app/gaussian_splatting/submodules/simple-knn/simple_knn /app/gaussian_splatting/

# Create an improved helper script for running with xvfb
RUN printf '#!/bin/bash\n\
# Set the appropriate EGL configuration\n\
export __EGL_VENDOR_LIBRARY_DIRS=/usr/share/glvnd/egl_vendor.d\n\
export __GLX_VENDOR_LIBRARY_NAME=nvidia\n\
export __GL_SYNC_TO_VBLANK=0\n\
\n\
# Run with a better X configuration\n\
xvfb-run -a -s "-screen 0 1280x1024x24 +extension GLX +render -noreset" "$@"\n' > /app/run_with_xvfb.sh && \
    chmod +x /app/run_with_xvfb.sh

# Create entrypoint script
RUN printf '#!/bin/bash\nsource /opt/conda/etc/profile.d/conda.sh\nconda activate sugar\n\n# Execute any command passed to docker run\nexec "$@"\n' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

EXPOSE 3000
EXPOSE 5173
# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
