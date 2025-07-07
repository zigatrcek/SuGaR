#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate sugar

# Execute any command passed to docker run
exec "$@"
