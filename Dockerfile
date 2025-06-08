# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (since the base image has conda but we want to ensure it's available)
RUN conda --version || (wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh)

# Ensure conda is in PATH
ENV PATH=/opt/conda/bin:$PATH

# Initialize conda for bash
RUN conda init bash

# Set working directory
WORKDIR /workspace/SuGaR

# Copy project files
COPY . .

# Run the automated installation script
RUN python install.py

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "sugar", "/bin/bash", "-c"]


# Create output directories
RUN mkdir -p output

# Set up entrypoint script
RUN echo '#!/bin/bash\n\
conda activate sugar\n\
exec "$@"' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

# Expose port for the viewer
EXPOSE 3000

# Set the default entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["bash"]
