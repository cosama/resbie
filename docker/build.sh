#!/usr/bin/env bash
# Build the resbie offline image from a minimal context (core/, which holds
# the vendored RESPLE and BIEVR sources) so the repository's datasets never
# enter the context.
#
#   docker/build.sh [image-tag]
#
# Default tag: ghcr.io/cosama/resbie:latest.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
tag="${1:-ghcr.io/cosama/resbie:latest}"
if [ "$#" -gt 0 ]; then shift; fi
engine="${CONTAINER_ENGINE:-docker}"

tar -C "${repo_root}" -c \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.git' \
    --exclude='core/build' \
    --exclude='core/.venv' \
    --exclude='core/.pytest_cache' \
    --transform='s,^docker/Dockerfile$,Dockerfile,' \
    docker/Dockerfile \
    core \
  | "${engine}" build -t "${tag}" "$@" -
