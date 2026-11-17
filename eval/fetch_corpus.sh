#!/usr/bin/env bash
# Downloads the SEC 10-K passage corpus (1,880 passages, ~0.5 MB) and verifies its SHA-256.
# Source: becloudready/labagent-datasets release sec-10k-2026.10; text derived from SEC EDGAR filings.
set -euo pipefail
cd "$(dirname "$0")" && mkdir -p data && cd data
base=https://github.com/becloudready/labagent-datasets/releases/download/sec-10k-2026.10
curl -sSLO "$base/sec-10k-2026.10.chunks.jsonl.gz"
echo "4a327bfab2eee9c466b297ae4a7f2480c40188bc20809ecd270794fc1294775e  sec-10k-2026.10.chunks.jsonl.gz" | sha256sum -c -
