#!/bin/bash

CUDA_ENABLED=${CUDA_ENABLED:-true}
DEVICE=""

if [ "${CUDA_ENABLED}" != "true" ]; then
    DEVICE="--device cpu"
fi

# Launch FastAPI server with Uvicorn
exec uvicorn tools.custom_api.endpoints:app --host 0.0.0.0 --port 8080 ${DEVICE}
