#!/usr/bin/env bash
#
# pack.sh - Archive ../bge-m3 and split it into <100MB
# chunks that can be committed to GitHub WITHOUT Git LFS, then reconstructed
# later by download.sh.
#
# Run this from inside the bge-m3_dist/ directory whenever
# the model contents change.
#
set -euo pipefail

# ---- config -------------------------------------------------------------
MODEL_DIR_NAME="bge-m3"
SRC_DIR="../${MODEL_DIR_NAME}"
ARCHIVE="${MODEL_DIR_NAME}.tar.gz"
CHUNK_PREFIX="${ARCHIVE}.part"
CHUNK_SIZE="90m"          # under GitHub's 100MB no-LFS hard limit
MANIFEST="manifest.txt"
# -------------------------------------------------------------------------

cd "$(dirname "$0")"

if [[ ! -d "$SRC_DIR" ]]; then
  echo "ERROR: source dir not found: $SRC_DIR" >&2
  exit 1
fi

# sha256 helper (macOS uses shasum, linux usually sha256sum)
sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

echo ">> Cleaning previous chunks..."
rm -f "${CHUNK_PREFIX}"* "$ARCHIVE" "$MANIFEST"

echo ">> Creating archive ${ARCHIVE} from ${SRC_DIR} ..."
# Exclude junk / caches so the archive is deterministic and lean.
tar \
  --exclude='.DS_Store' \
  --exclude='.cache' \
  -czf "$ARCHIVE" -C "$SRC_DIR/.." "$MODEL_DIR_NAME"

ARCHIVE_SHA="$(sha256 "$ARCHIVE")"
ARCHIVE_BYTES="$(wc -c < "$ARCHIVE" | tr -d ' ')"

echo ">> Splitting into ${CHUNK_SIZE} chunks..."
# -d would give numeric suffixes but isn't on all BSD splits; default aa,ab,...
split -b "$CHUNK_SIZE" "$ARCHIVE" "$CHUNK_PREFIX"

echo ">> Writing ${MANIFEST} ..."
{
  echo "# manifest for ${MODEL_DIR_NAME}"
  echo "# format: <sha256>  <filename>"
  echo "archive_name ${ARCHIVE}"
  echo "archive_sha256 ${ARCHIVE_SHA}"
  echo "archive_bytes ${ARCHIVE_BYTES}"
  echo "chunk_prefix ${CHUNK_PREFIX}"
  echo "# --- chunks (in order) ---"
  for f in "${CHUNK_PREFIX}"*; do
    echo "$(sha256 "$f")  ${f}"
  done
} > "$MANIFEST"

# The full archive is large + redundant with the chunks; remove it.
rm -f "$ARCHIVE"

echo
echo ">> Done. Chunks:"
ls -lh "${CHUNK_PREFIX}"*
echo
echo "Next: ./upload.sh    # uploads chunks + manifest as GitHub Release assets"
