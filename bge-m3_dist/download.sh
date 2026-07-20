#!/usr/bin/env bash
#
# download.sh - Download the split chunks of bge-m3 from the
# GitHub Release using PUBLIC URLs only (no gh, no auth), verify checksums,
# reconstruct the archive and extract it.
#
# Modes:
#   ./download.sh [DEST]            fetch assets from public release URLs (curl)
#   ./download.sh --local [DEST]    use chunks already next to this script
#
# NOTE: public URLs require the repo/release to be PUBLIC.
#
set -euo pipefail

# ---- config -------------------------------------------------------------
GH_REPO="kodiakllc/kdk-mlx"
RELEASE_TAG="model-bge-m3"
MODEL_DIR_NAME="bge-m3"
ARCHIVE="${MODEL_DIR_NAME}.tar.gz"
MANIFEST="manifest.txt"
BASE_URL="https://github.com/${GH_REPO}/releases/download/${RELEASE_TAG}"
# -------------------------------------------------------------------------

# ---- pretty status helpers ---------------------------------------------
if [[ -t 1 ]]; then
  C_RST=$'\033[0m'; C_GRN=$'\033[32m'; C_BLU=$'\033[34m'
  C_YLW=$'\033[33m'; C_RED=$'\033[31m'; C_DIM=$'\033[2m'; C_BLD=$'\033[1m'
else
  C_RST=; C_GRN=; C_BLU=; C_YLW=; C_RED=; C_DIM=; C_BLD=
fi
step() { printf "%s==>%s %s%s%s\n" "$C_BLU" "$C_RST" "$C_BLD" "$1" "$C_RST"; }
ok()   { printf "  %s✓%s %s\n" "$C_GRN" "$C_RST" "$1"; }
info() { printf "  %s•%s %s\n" "$C_DIM" "$C_RST" "$1"; }
err()  { printf "  %s✗%s %s\n" "$C_RED" "$C_RST" "$1" >&2; }
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

# fetch <asset-name> <out-path> [progress]
fetch() {
  local name="$1" out="$2" prog="${3:-}"
  if [[ $LOCAL -eq 1 ]]; then
    cp "$(dirname "$0")/${name}" "$out"
  elif [[ -n "$prog" ]]; then
    # visible progress bar for large chunks
    curl -fL --progress-bar "${BASE_URL}/${name}" -o "$out"
  else
    curl -fsSL "${BASE_URL}/${name}" -o "$out"
  fi
}

step "Fetching manifest"
if ! fetch "$MANIFEST" "$WORK/$MANIFEST"; then
  err "could not download ${MANIFEST} from ${BASE_URL}"
  err "is the repo/release public?"
  exit 1
fi
ok "manifest.txt"

ARCHIVE_SHA="$(awk '/^archive_sha256/{print $2}' "$WORK/$MANIFEST")"
# chunk lines: <sha256>  <name>.tar.gz.partXX  (skip comments/metadata)
CHUNK_MANIFEST="$(grep -E '\.tar\.gz\.part' "$WORK/$MANIFEST" | grep -vE '^#|chunk_prefix')"
NUM_CHUNKS="$(printf '%s\n' "$CHUNK_MANIFEST" | grep -c . || true)"

step "Downloading ${NUM_CHUNKS} chunks"
CHUNK_FILES=()
idx=0
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  idx=$((idx+1))
  want_sha="$(echo "$line" | awk '{print $1}')"
  fname="$(echo "$line" | awk '{print $2}')"
  printf "  %s[%d/%d]%s %s\n" "$C_YLW" "$idx" "$NUM_CHUNKS" "$C_RST" "$fname"
  fetch "$fname" "$WORK/$fname" progress
  got_sha="$(sha256 "$WORK/$fname")"
  if [[ "$got_sha" != "$want_sha" ]]; then
    err "checksum mismatch for $fname"
    err "  expected $want_sha"
    err "  got      $got_sha"
    exit 1
  fi
  ok "verified $fname"
  CHUNK_FILES+=("$WORK/$fname")
done <<< "$CHUNK_MANIFEST"

step "Reassembling archive"
# sort to guarantee partaa, partab, ... order regardless of fetch order
IFS=$'\n' CHUNK_FILES=($(sort <<<"${CHUNK_FILES[*]}")); unset IFS
cat "${CHUNK_FILES[@]}" > "$WORK/$ARCHIVE"
ok "joined ${#CHUNK_FILES[@]} chunks -> ${ARCHIVE}"

step "Verifying full archive checksum"
got="$(sha256 "$WORK/$ARCHIVE")"
if [[ "$got" != "$ARCHIVE_SHA" ]]; then
  err "reassembled archive checksum mismatch"
  err "  expected $ARCHIVE_SHA"
  err "  got      $got"
  exit 1
fi
ok "sha256 OK"

step "Extracting into ${DEST_DIR}"
mkdir -p "$DEST_DIR"
tar -xzf "$WORK/$ARCHIVE" -C "$DEST_DIR"
ok "extracted"

printf "\n%s✓ Done.%s Model available at: %s%s/%s%s\n" \
  "$C_GRN" "$C_RST" "$C_BLD" "${DEST_DIR%/}" "$MODEL_DIR_NAME" "$C_RST"
