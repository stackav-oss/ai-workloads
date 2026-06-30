#!/bin/bash
set -e

# Build lidar_vae Docker image for Volt (ARM64).
# Run from this repo root: bash build_volt.sh --local

cd "$(dirname "$0")"

GH_USER=$(git config github.user 2>/dev/null || true)
if [[ -z "$GH_USER" ]]; then
  GH_USER="$USER"
fi

LOCAL=0
for arg in "$@"; do
  [[ "$arg" == "--local" ]] && LOCAL=1
done

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
GHCR_TAG="ghcr.io/${GH_USER}/lidar_vae:${TIMESTAMP}"
ECR_TAG="577004484676.dkr.ecr.us-east-1.amazonaws.com/cache/${GHCR_TAG}"

if [[ "$LOCAL" == "1" ]]; then
  echo "Building locally for linux/arm64: $GHCR_TAG"

  VOLT_CONFIG="$HOME/.volt/config"
  if [[ -f "$VOLT_CONFIG" ]]; then
    export AWS_ACCESS_KEY_ID=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.volt/config')))['accessKeyId'])")
    export AWS_SECRET_ACCESS_KEY=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.volt/config')))['secretAccessKey'])")
    export AWS_SESSION_TOKEN=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.volt/config')))['sessionToken'])")
    export AWS_REGION=us-east-1
    unset AWS_PROFILE
    echo "Attempting ECR login..."
    if ! aws ecr get-login-password --region us-east-1 \
      | docker login --username AWS --password-stdin 577004484676.dkr.ecr.us-east-1.amazonaws.com; then
      echo "WARNING: ECR login failed; continuing local build without registry auth."
    fi
  else
    echo "WARNING: ~/.volt/config not found; continuing local build without registry auth."
  fi

  docker run --privileged --rm tonistiigi/binfmt --install arm64 2>/dev/null || true

  docker buildx build \
    --platform linux/arm64 \
    --load \
    -f Dockerfile.volt \
    -t "$GHCR_TAG" \
    .

  echo ""
  echo "Local build complete. Image tagged: $GHCR_TAG"
  echo ""
  echo "To mirror to ECR, run:"
  echo "  AWS_PROFILE=magic ./mirror_to_ecr.sh --local $GHCR_TAG"
  echo ""
  echo "Then update autoencoder.yaml with:"
  echo "  image: $ECR_TAG"
  exit 0
fi

echo "Usage: $0 --local"
