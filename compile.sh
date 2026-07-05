#!/bin/bash
set -euo pipefail

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

# Use half of available CPU cores for Cargo builds to reduce OOM risk.
# If CARGO_BUILD_JOBS is already set, respect that value.
if [[ -z "${CARGO_BUILD_JOBS:-}" ]]; then
    CORES="$(nproc)"
    HALF_CORES="$(( CORES / 2 ))"

    # Ensure at least 1 job is used.
    if (( HALF_CORES < 1 )); then
        HALF_CORES=1
    fi

    export CARGO_BUILD_JOBS="$HALF_CORES"
fi

echo "Building and installing Zed fork (using ${CARGO_BUILD_JOBS} parallel job(s), half of available cores)..."
./script/install-linux

echo "Zed fork built and installed successfully!"
echo "Desktop entry: ~/.local/share/applications/dev.zed.Zed-Dev.desktop"
echo "Binary link: ~/.local/bin/zed"
