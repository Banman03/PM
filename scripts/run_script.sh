#!/bin/bash
set -e

make

# Example asset_id for a popular market
# You can replace this with any valid Polymarket asset ID
ASSET_ID="${1:-21742633143463906290569050155826241533067272736897614950488156847949938836455}"

./build/main "$ASSET_ID"
