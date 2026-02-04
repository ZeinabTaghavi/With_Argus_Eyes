# Data directory

Place raw/interim/processed datasets here.

## Tracked processed dataset
`data/processed/8_Emb_Rank/` is versioned in git because it is part of the published dataset. This can make the repo large.

If you want to avoid downloading the dataset, use sparse checkout (see `README.md`).

## Integrity verification
A hash manifest is provided at:
- `data/processed/8_Emb_Rank/SHA256SUMS`

Verify the dataset files with:
```bash
shasum -a 256 data/processed/8_Emb_Rank/*.jsonl | shasum -a 256 -c data/processed/8_Emb_Rank/SHA256SUMS
```
