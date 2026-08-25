#!/bin/bash
# Run the full foilpolars pipeline (sweep, then save) for a config.
# Usage: ./run.sh [small|mhk]   (no arg -> configs/config.yaml)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

# Map the optional tag arg to its config file
tag="${1:-}"
case "$tag" in
    "") config="configs/config.yaml" ;;
    small) config="configs/config_small.yaml" ;;
    mhk) config="configs/config_mhk.yaml" ;;
    *)
        echo "Usage: $0 [small|mhk]" >&2
        exit 1
        ;;
esac

if [[ ! -f "$config" ]]; then
    echo "Config not found: $config" >&2
    exit 1
fi

echo "[run.sh] using $config"
uv run foilpolars sweep --config "$config"
uv run foilpolars save --config "$config"
