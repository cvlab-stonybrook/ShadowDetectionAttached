#!/usr/bin/env bash
set -euo pipefail

python test.py --config configs/test.yaml "$@"
