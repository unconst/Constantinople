#!/bin/bash
# Build, tag, and push inference subnet Docker images.
# Watchtower on the target host auto-pulls within 5 minutes.
#
# Usage:
#   bash deploy_docker.sh                 # build + push both
#   bash deploy_docker.sh --validator     # validator only
#   bash deploy_docker.sh --miner         # miner only
#   bash deploy_docker.sh --no-push       # build only, skip push

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

PREFIX="${IMAGE_PREFIX:-thebes1618/inference-subnet}"
DO_PUSH=true
BUILD_VALIDATOR=true
BUILD_MINER=true
GPU_ARG=0

for arg in "$@"; do
    case "$arg" in
        --validator) BUILD_MINER=false ;;
        --miner)     BUILD_VALIDATOR=false ;;
        --no-push)   DO_PUSH=false ;;
        --gpu)       GPU_ARG=1 ;;
    esac
done

echo "=== Inference Subnet Docker Deploy ==="
echo "Prefix: $PREFIX"
echo "Push:   $DO_PUSH"

if $BUILD_VALIDATOR; then
    echo ""
    echo "--- Building validator (GPU=$GPU_ARG) ---"
    docker build -f Dockerfile.validator --build-arg GPU=$GPU_ARG \
        -t "${PREFIX}-validator:latest" .
    echo "Validator image: ${PREFIX}-validator:latest"
fi

if $BUILD_MINER; then
    echo ""
    echo "--- Building miner ---"
    docker build -f Dockerfile.miner \
        -t "${PREFIX}-miner:latest" .
    echo "Miner image: ${PREFIX}-miner:latest"
fi

if $DO_PUSH; then
    if $BUILD_VALIDATOR; then
        echo ""
        echo "--- Pushing validator ---"
        docker push "${PREFIX}-validator:latest"
    fi
    if $BUILD_MINER; then
        echo ""
        echo "--- Pushing miner ---"
        docker push "${PREFIX}-miner:latest"
    fi
    echo ""
    echo "=== Push complete. Watchtower will pick up new images within 5 minutes. ==="
else
    echo ""
    echo "=== Build complete (push skipped). ==="
fi
