#!/usr/bin/env bash
#
# download.sh - Download the split chunks of multi-qa-MiniLM-L6-cos-v1 from a
# GitHub Release, verify their checksums, reconstruct the archive and extract.
#
# Modes:
#   ./download.sh [DEST]            fetch assets from the GitHub Release (uses gh)
#   ./download.sh --local [DEST]    use chunks already next to this script
#
# Fetch order for the release: prefers `gh` (works for private repos); falls
# back to public release-download URLs via curl if gh is unavailable.
#
set -euo pipefail

# ---- config -------------------------------------------------------------
GH_REPO="kodiakllc/kdk-mlx"
RELEASE_TAG="model-multi-qa-MiniLM-L6-cos-v1"
MODEL_DIR_NAME="multi-qa-MiniLM-L6-cos-v1"
ARCHIVE="${MODEL_DIR_NAME}.tar.gz"
MANIFEST="manifest.txt"
# -------------------------------------------------------------------------

LOCAL=0
DEST_DIR="."
if [[ "${1:-}" == "--local" ]]; then
  LOCAL=1; DEST_DIR="${2:-.}"
else
  DEST_DIR="${1:-.}"
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

# fetch <asset-name> <out-path>
fetch() {
  local name="$1" out="$2"
  if [[ $LOCAL -eq 1 ]]; then
    cp "$(dirname "$0")/${name}" "$out"
  elif command -v gh >/dev/null 2>&1; then
    echo "   .. ${name} (gh)"
    gh release download "$RELEASE_TAG" --repo "$GH_REPO" \
      --pattern "$name" --dir "$(dirname "$out")" --clobber
  else
    echo "   .. ${name} (curl, public only)"
    curl -fsSL \
      "https://github.com/${GH_REPO}/releases/download/${RELEASE_TAG}/${name}" \
      -o "$out"
  fi
}

echo ">> Getting manifest..."
fetch "$MANIFEST" "$WORK/$MANIFEST"

ARCHIVE_SHA="$(awk '/^archive_sha256/{print $2}' "$WORK/$MANIFEST")"
# chunk lines: <sha256>  <name>.tar.gz.partXX  (skip comments/metadata)
CHUNK_MANIFEST="$(grep -E '\.tar\.gz\.part' "$WORK/$MANIFEST" | grep -vE '^#|chunk_prefix')"

NUM_CHUNKS="$(printf '%s\n' "$CHUNK_MANIFEST" | grep -c . || true)"
echo ">> Downloading ${NUM_CHUNKS} chunks..."
CHUNK_FILES=()
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  want_sha="$(echo "$line" | awk '{print $1}')"
  fname="$(echo "$line" | awk '{print $2}')"
  fetch "$fname" "$WORK/$fname"
  got_sha="$(sha256 "$WORK/$fname")"
  if [[ "$got_sha" != "$want_sha" ]]; then
    echo "ERROR: checksum mismatch for $fname" >&2
    echo "  expected $want_sha" >&2
    echo "  got      $got_sha" >&2
    exit 1
  fi
  CHUNK_FILES+=("$WORK/$fname")
done <<< "$CHUNK_MANIFEST"

echo ">> Reassembling archive..."
# sort to guarantee partaa, partab, ... order regardless of fetch order
IFS=$'\n' CHUNK_FILES=($(sort <<<"${CHUNK_FILES[*]}")); unset IFS
cat "${CHUNK_FILES[@]}" > "$WORK/$ARCHIVE"

echo ">> Verifying full archive checksum..."
got="$(sha256 "$WORK/$ARCHIVE")"
if [[ "$got" != "$ARCHIVE_SHA" ]]; then
  echo "ERROR: reassembled archive checksum mismatch" >&2
  echo "  expected $ARCHIVE_SHA" >&2
  echo "  got      $got" >&2
  exit 1
fi

echo ">> Extracting into ${DEST_DIR} ..."
mkdir -p "$DEST_DIR"
tar -xzf "$WORK/$ARCHIVE" -C "$DEST_DIR"

echo
echo ">> Done. Model available at: ${DEST_DIR%/}/${MODEL_DIR_NAME}"
