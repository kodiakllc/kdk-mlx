#!/usr/bin/env bash
#
# upload.sh - Upload the split chunks + manifest of bge-m3
# as GitHub Release assets (NOT committed to git, NOT Git LFS).
#
# Requires: gh (GitHub CLI), authenticated via `gh auth login`.
# (gh is only needed to PUBLISH; download.sh fetches via public URLs.)
#
set -euo pipefail

# ---- config -------------------------------------------------------------
GH_REPO="kodiakllc/kdk-mlx"
RELEASE_TAG="model-bge-m3"
RELEASE_TITLE="bge-m3 (chunked model assets)"
MODEL_DIR_NAME="bge-m3"
CHUNK_PREFIX="${MODEL_DIR_NAME}.tar.gz.part"
MANIFEST="manifest.txt"
# -------------------------------------------------------------------------

# ---- pretty status helpers ---------------------------------------------
if [[ -t 1 ]]; then
  C_RST=$'\033[0m'; C_GRN=$'\033[32m'; C_BLU=$'\033[34m'
  C_RED=$'\033[31m'; C_DIM=$'\033[2m'; C_BLD=$'\033[1m'
else
  C_RST=; C_GRN=; C_BLU=; C_RED=; C_DIM=; C_BLD=
fi
step() { printf "%s==>%s %s%s%s\n" "$C_BLU" "$C_RST" "$C_BLD" "$1" "$C_RST"; }
ok()   { printf "  %s✓%s %s\n" "$C_GRN" "$C_RST" "$1"; }
info() { printf "  %s•%s %s\n" "$C_DIM" "$C_RST" "$1"; }
err()  { printf "  %s✗%s %s\n" "$C_RED" "$C_RST" "$1" >&2; }
# -------------------------------------------------------------------------

cd "$(dirname "$0")"

step "Preflight"
if ! command -v gh >/dev/null 2>&1; then
  err "gh (GitHub CLI) not found. Install it and run 'gh auth login'."
  exit 1
fi
ok "gh found"
if ! gh auth status >/dev/null 2>&1; then
  err "gh not authenticated. Run: gh auth login"
  exit 1
fi
ok "gh authenticated"
if ! compgen -G "${CHUNK_PREFIX}*" >/dev/null; then
  err "no chunks found. Run ./pack.sh first."
  exit 1
fi
[[ -f "$MANIFEST" ]] || { err "$MANIFEST missing. Run ./pack.sh."; exit 1; }
CHUNKS=( ${CHUNK_PREFIX}* )
ok "found ${#CHUNKS[@]} chunks + manifest"

step "Ensuring release ${RELEASE_TAG}"
if ! gh release view "$RELEASE_TAG" --repo "$GH_REPO" >/dev/null 2>&1; then
  gh release create "$RELEASE_TAG" --repo "$GH_REPO" \
    --title "$RELEASE_TITLE" \
    --notes "Chunked distribution of ${MODEL_DIR_NAME}. Reconstruct with download.sh." >/dev/null
  ok "release created"
else
  info "release already exists; assets will be overwritten"
fi

step "Uploading ${#CHUNKS[@]} chunks + manifest"
total=$(( ${#CHUNKS[@]} + 1 )); i=0
for f in "$MANIFEST" "${CHUNKS[@]}"; do
  i=$((i+1))
  printf "  %s[%d/%d]%s %s\n" "$C_DIM" "$i" "$total" "$C_RST" "$f"
  gh release upload "$RELEASE_TAG" --repo "$GH_REPO" --clobber "$f" >/dev/null
  ok "uploaded $f"
done

printf "\n%s✓ Done.%s Assets at: %shttps://github.com/%s/releases/tag/%s%s\n" \
  "$C_GRN" "$C_RST" "$C_BLD" "$GH_REPO" "$RELEASE_TAG" "$C_RST"
