#!/bin/bash
# Build and push Constantinople Docker images to Docker Hub
#
# Usage:
#   ./build-and-push.sh                    # Build and push both images
#   ./build-and-push.sh --validator-only   # Only validator
#   ./build-and-push.sh --miner-only       # Only miner
#   ./build-and-push.sh --local            # Build locally, don't push
#
# Requires: DOCKER_USERNAME and DOCKER_TOKEN env vars for Docker Hub push

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
REPO="${DOCKER_REPO:-${DOCKER_USERNAME:-unconst}/constantinople}"
TAG="${TAG:-latest}"
VALIDATOR_IMAGE="${REPO}-validator:${TAG}"
MINER_IMAGE="${REPO}-miner:${TAG}"

# Parse args
BUILD_VALIDATOR=true
BUILD_MINER=true
PUSH=true

for arg in "$@"; do
    case "$arg" in
        --validator-only) BUILD_MINER=false ;;
        --miner-only) BUILD_VALIDATOR=false ;;
        --local) PUSH=false ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# Login to Docker Hub
if [ "$PUSH" = true ]; then
    if [ -z "${DOCKER_TOKEN:-}" ]; then
        echo "ERROR: DOCKER_TOKEN not set. Cannot push."
        exit 1
    fi
    echo "Logging in to Docker Hub..."
    docker login -u "${DOCKER_USERNAME:-unconst}" -p "$DOCKER_TOKEN"
fi

# Build validator
if [ "$BUILD_VALIDATOR" = true ]; then
    echo "Building validator: ${VALIDATOR_IMAGE}"
    docker build -f Dockerfile.validator -t "$VALIDATOR_IMAGE" .
    if [ "$PUSH" = true ]; then
        echo "Pushing validator..."
        docker push "$VALIDATOR_IMAGE"
    fi
    echo "Validator done: ${VALIDATOR_IMAGE}"
fi

# Build miner
if [ "$BUILD_MINER" = true ]; then
    echo "Building miner: ${MINER_IMAGE}"
    docker build -f Dockerfile.miner -t "$MINER_IMAGE" .
    if [ "$PUSH" = true ]; then
        echo "Pushing miner..."
        docker push "$MINER_IMAGE"
    fi
    echo "Miner done: ${MINER_IMAGE}"
fi

echo ""
echo "=== Build Complete ==="
echo "Validator: ${VALIDATOR_IMAGE}"
echo "Miner:     ${MINER_IMAGE}"
if [ "$PUSH" = true ]; then
    echo "Images pushed to Docker Hub"
    echo "Watchtower will auto-pull new images every 5 minutes."
fi
