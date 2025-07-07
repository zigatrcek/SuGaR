# SuGaR Dev Container

This directory contains the VS Code Dev Container configuration for the SuGaR project.

## Prerequisites

1. **Docker** with GPU support (NVIDIA Container Toolkit)
2. **VS Code** with the Dev Containers extension
3. **NVIDIA GPU** with CUDA support

## Setup

1. Open this project in VS Code
2. When prompted, click "Reopen in Container" or use the command palette (`Ctrl+Shift+P`) and select "Dev Containers: Reopen in Container"
3. The container will build automatically using the Dockerfile in the root directory

## Features

- **Pre-configured Python environment** with the `sugar` conda environment
- **GPU support** enabled with `--gpus=all`
- **Port forwarding** for the SuGaR viewer (port 3000)
- **Volume mounts** for `data/` and `output/` directories
- **VS Code extensions** for Python, Jupyter, C++, and more

## Usage

Once the container is running:

1. The terminal will automatically activate the `sugar` conda environment
2. You can run SuGaR commands directly:
   ```bash
   python train_full_pipeline.py -s ./data/your_scene -r dn_consistency --high_poly True --export_obj True
   ```
3. Access the viewer at `http://localhost:3000` when running viewer commands
4. Your data and output directories are mounted from the host system

## Directory Structure

- `data/` - Mount point for your input datasets (create this directory on your host)
- `output/` - Mount point for generated outputs (create this directory on your host)

## Troubleshooting

- **GPU not detected**: Ensure NVIDIA Container Toolkit is installed and Docker has GPU access
- **Build fails**: Check that all submodules are properly initialized with `git submodule update --init --recursive`
- **Port conflicts**: Change the port mapping in `devcontainer.json` if port 3000 is already in use
