#!/usr/bin/env bash
set -euo pipefail

# Disable AWS CLI pager so aws commands do not open an interactive
# pager (e.g., requiring 'q' to exit) when run from this script.
export AWS_PAGER=""
export AWS_CLI_PAGER=""

# Mirror a container image into ECR in account 577004484676 (default region us-east-1).
# The ECR repository name is derived from the full source image path (minus tag),
# prefixed with "cache/".
# Example:
#   ghcr.io/mrhuff/fp4-training:v9 -> cache/ghcr.io/mrhuff/fp4-training
#
# Usage:
#   mirror_to_ecr.sh [--local] <source_image> [tag]
#
# Examples:
#   mirror_to_ecr.sh ghcr.io/mrhuff/fp4-training:v9
#   mirror_to_ecr.sh ghcr.io/owner/image:tag new-tag
#   mirror_to_ecr.sh --local alpine:3.19
#
# Environment overrides:
#   AWS_ACCOUNT_ID  (default: 577004484676)
#   AWS_REGION      (default: us-east-1)

AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-577004484676}
AWS_REGION=${AWS_REGION:-us-east-1}

CONFIG_FILE="${HOME}/.volt/config"

log() {
  echo "[mirror_to_ecr] $*"
}

usage() {
  log "Usage: $0 [--local] <source_image> [tag]" >&2
  exit 1
}

# Ensure AWS credentials from ~/.volt/config are available only for the
# lifetime of this script.
cleanup() {
  echo "Cleaning up AWS credentials from environment..."
  unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN || true
}
trap cleanup EXIT

parse_args() {
  LOCAL_MODE="false"

  # Support both:
  #   mirror_to_ecr.sh --local <source_image> [tag]
  # Strip out any --local flag from the arguments, regardless of position.
  POSITIONAL_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --local)
        LOCAL_MODE="true"
        shift
        ;;
      *)
        POSITIONAL_ARGS+=("$1")
        shift
        ;;
    esac
  done

  set -- "${POSITIONAL_ARGS[@]:-}"

  if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage
  fi

  SOURCE_IMAGE="$1"      # e.g. ghcr.io/mrhuff/fp4-training:v9

  # Derive ECR repository name from the full source image path (minus tag),
  # and prefix it with "cache/" so that
  #   ghcr.io/mrhuff/fp4-training:v9 -> cache/ghcr.io/mrhuff/fp4-training
  SOURCE_NO_TAG="${SOURCE_IMAGE%%:*}"
  CACHE_PREFIX="cache"
  ECR_REPO_NAME="${CACHE_PREFIX}/${SOURCE_NO_TAG}"

  if [[ $# -eq 2 ]]; then
    TARGET_TAG="$2"
  else
    # Derive tag from source image (part after last colon); default to 'latest' if none.
    if [[ "$SOURCE_IMAGE" == *:* ]]; then
      TARGET_TAG="${SOURCE_IMAGE##*:}"
    else
      TARGET_TAG="latest"
    fi
  fi

  ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  TARGET_IMAGE="${ECR_REGISTRY}/${ECR_REPO_NAME}:${TARGET_TAG}"
}

load_aws_credentials() {
  if [[ ! -f "${CONFIG_FILE}" ]]; then
    log "ERROR: Missing credentials file ${CONFIG_FILE}. Please run 'volt --login'." >&2
    exit 1
  fi

  if ! command -v jq >/dev/null 2>&1; then
    log "ERROR: 'jq' is required but not found on PATH." >&2
    exit 1
  fi

  ACCESS_KEY_ID=$(jq -r '.accessKeyId // empty' "${CONFIG_FILE}")
  SECRET_ACCESS_KEY=$(jq -r '.secretAccessKey // empty' "${CONFIG_FILE}")
  SESSION_TOKEN=$(jq -r '.sessionToken // empty' "${CONFIG_FILE}")
  EXPIRES_AT=$(jq -r '.expiresAt // empty' "${CONFIG_FILE}")

  if [[ -z "${ACCESS_KEY_ID}" || -z "${SECRET_ACCESS_KEY}" || -z "${SESSION_TOKEN}" || -z "${EXPIRES_AT}" ]]; then
    log "ERROR: Incomplete credentials in ${CONFIG_FILE}. Please run 'volt --login'." >&2
    exit 1
  fi

  if ! command -v python3 >/dev/null 2>&1; then
    log "ERROR: 'python3' is required to validate credential expiry." >&2
    exit 1
  fi

  # Check whether the credentials have expired using Python's ISO-8601 parser.
  if ! python3 - "${EXPIRES_AT}" << 'PY'
import sys
from datetime import datetime

expires_at = sys.argv[1]

try:
    exp = datetime.fromisoformat(expires_at)
except Exception:
    # Unable to parse the expiry timestamp
    sys.exit(2)

now = datetime.now(exp.tzinfo)

if now >= exp:
    # Expired
    sys.exit(1)

# Still valid
sys.exit(0)
PY
  then
    rc=$?
    if [[ ${rc} -eq 1 ]]; then
      log "ERROR: Credentials in ${CONFIG_FILE} have expired. Please run 'volt --login'." >&2
      exit 1
    elif [[ ${rc} -eq 2 ]]; then
      log "ERROR: Could not parse 'expiresAt' in ${CONFIG_FILE}. Please run 'volt --login'." >&2
      exit 1
    fi
  fi

  export AWS_ACCESS_KEY_ID="${ACCESS_KEY_ID}"
  export AWS_SECRET_ACCESS_KEY="${SECRET_ACCESS_KEY}"
  export AWS_SESSION_TOKEN="${SESSION_TOKEN}"
}

validate_account_id() {
  # Verify that the configured AWS_ACCOUNT_ID matches the account of the
  # currently active AWS credentials. This avoids creating the repository in
  # one account while pushing to another registry.
  ACTUAL_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
  if [[ -n "${ACTUAL_ACCOUNT_ID}" && "${ACTUAL_ACCOUNT_ID}" != "${AWS_ACCOUNT_ID}" ]]; then
    log "ERROR: AWS_ACCOUNT_ID (${AWS_ACCOUNT_ID}) does not match current credentials account (${ACTUAL_ACCOUNT_ID})." >&2
    log "Please set AWS_ACCOUNT_ID correctly or switch AWS profile/SSO session." >&2
    exit 1
  fi
}

ensure_ecr_repository() {
  log "Ensuring ECR repository ${ECR_REPO_NAME} exists in ${AWS_REGION}..."
  if ! aws ecr describe-repositories \
    --repository-names "${ECR_REPO_NAME}" \
    --region "${AWS_REGION}" >/dev/null 2>&1; then
    log "Creating ECR repository ${ECR_REPO_NAME}..."
    aws ecr create-repository \
      --repository-name "${ECR_REPO_NAME}" \
      --region "${AWS_REGION}" >/dev/null

    log "Applying 30-day lifecycle policy to ${ECR_REPO_NAME}..."
    aws ecr put-lifecycle-policy \
      --repository-name "${ECR_REPO_NAME}" \
      --region "${AWS_REGION}" \
      --lifecycle-policy-text '{
        "rules": [
          {
            "rulePriority": 1,
            "description": "Expire images older than 30 days",
            "selection": {
              "tagStatus": "any",
              "countType": "sinceImagePushed",
              "countUnit": "days",
              "countNumber": 30
            },
            "action": { "type": "expire" }
          }
        ]
      }' >/dev/null
  else
    log "ECR repository ${ECR_REPO_NAME} already exists."
  fi
}

mirror_image() {
  log "Source image : ${SOURCE_IMAGE}"
  log "Target image : ${TARGET_IMAGE}"

  log "Pulling source image..."
  if [[ "${LOCAL_MODE}" != "true" ]]; then
    docker pull "${SOURCE_IMAGE}"
  else
    log "--local enabled, using existing local image; skipping pull."
  fi

  ensure_ecr_repository

  log "Logging in to ECR ${ECR_REGISTRY}..."
  aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login \
        --username AWS \
        --password-stdin "${ECR_REGISTRY}"

  log "Tagging image as ${TARGET_IMAGE}..."
  docker tag "${SOURCE_IMAGE}" "${TARGET_IMAGE}"

  log "Pushing image to ECR..."
  docker push "${TARGET_IMAGE}"

  log "Done. Pushed ${TARGET_IMAGE}"
}

main() {
  parse_args "$@"
  load_aws_credentials
  validate_account_id
  mirror_image
}

main "$@"