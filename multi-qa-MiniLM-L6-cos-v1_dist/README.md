# multi-qa-MiniLM-L6-cos-v1 — chunked distribution via GitHub Releases (no Git LFS)

The full model (`../multi-qa-MiniLM-L6-cos-v1`, ~845MB) is shipped as a gzip
tarball split into **90MB chunks** and uploaded as **GitHub Release assets** —
not committed to git, and not Git LFS. Only the small scripts + `manifest.txt`
live in the repo.

- Repo:    `kodiakllc/kdk-mlx`
- Release: tag `model-multi-qa-MiniLM-L6-cos-v1`

## Files (committed to repo)
- `pack.sh`      — (re)build chunks from `../multi-qa-MiniLM-L6-cos-v1`
- `upload.sh`    — upload chunks + manifest as Release assets (needs `gh`)
- `download.sh`  — download assets from the Release, verify, reassemble, extract
- `manifest.txt` — sha256 of every chunk + the full archive

## Files NOT committed (git-ignored, live in the Release)
- `multi-qa-MiniLM-L6-cos-v1.tar.gz.part??`

## Publish (one-time per model update)
```bash
gh auth login                 # if not already authenticated
./pack.sh                     # build chunks + manifest
git add manifest.txt pack.sh upload.sh download.sh README.md
git commit -m "Update multi-qa-MiniLM-L6-cos-v1 dist scripts"
./upload.sh                   # creates/updates the release + uploads chunks
```

## Download + reconstruct
```bash
# from a repo checkout:
./download.sh /path/to/dest          # -> /path/to/dest/multi-qa-MiniLM-L6-cos-v1

# standalone (grab just the script, it pulls the rest from the release):
curl -fsSL https://raw.githubusercontent.com/kodiakllc/kdk-mlx/refs/heads/develop/phase-2/multi-qa-MiniLM-L6-cos-v1_dist/download.sh -o download.sh
bash download.sh /path/to/dest
```
- Uses `gh release download` when `gh` is available (works for private repos).
- Falls back to public `releases/download/<tag>/<asset>` URLs via `curl` if
  `gh` isn't installed (public repos only).
- `./download.sh --local /path/to/dest` reconstructs from chunks already next
  to the script (offline).

## Notes
- Chunks are 90MB, well under any per-asset limit; release assets have no
  practical total-size cap and never bloat git history.
- Checksums are verified per-chunk and for the full reassembled archive.
