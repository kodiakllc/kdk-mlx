#!/usr/bin/env bash
#
# upload.sh - Upload the split chunks + manifest of multi-qa-MiniLM-L6-cos-v1
# as GitHub Release assets (NOT committed to git, NOT Git LFS).
#
# Requires: gh (GitHub CLI), authenticated via `gh auth login`.
#
set -euo pipefail

# ---- config -------------------------------------------------------------
GH_REPO="kodiakllc/kdk-mlx"
RELEASE_TAG="model-multi-qa-MiniLM-L6-cos-v1"
RELEASE_TITLE="multi-qa-MiniLM-L6-cos-v1 (chunked model assets)"
MODEL_DIR_NAME="multi-qa-MiniLM-L6-cos-v1"
CHUNK_PREFIX="${MODEL_DIR_NAME}.tar.gz.part"
MANIFEST="manifest.txt"
# -------------------------------------------------------------------------

cd "$(dirname "$0")"

if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: gh (GitHub CLI) not found. Install it and run 'gh auth login'." >&2
  exit 1
fi

if ! compgen -G "${CHUNK_PREFIX}*" >/dev/null; then
  echo "ERROR: no chunks found. Run ./pack.sh first." >&2
  exit 1
fi
[[ -f "$MANIFEST" ]] || { echo "ERROR: $MANIFEST missing. Run ./pack.sh." >&2; exit 1; }

# Create the release if it doesn't exist yet.
if ! gh release view "$RELEASE_TAG" --repo "$GH_REPO" >/dev/null 2>&1; then
  echo ">> Creating release ${RELEASE_TAG} ..."
  gh release create "$RELEASE_TAG" --repo "$GH_REPO" \
    --title "$RELEASE_TITLE" \
    --notes "Chunked distribution of ${MODEL_DIR_NAME}. Reconstruct with download.sh."
else
  echo ">> Release ${RELEASE_TAG} already exists; uploading/overwriting assets..."
fi

echo ">> Uploading manifest + chunks (--clobber overwrites existing assets)..."
gh release upload "$RELEASE_TAG" --repo "$GH_REPO" --clobber \
  "$MANIFEST" ${CHUNK_PREFIX}*

echo
echo ">> Done. Assets available at:"
echo "   https://github.com/${GH_REPO}/releases/tag/${RELEASE_TAG}"
