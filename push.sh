#!/bin/bash
set -euo pipefail

HOSTS=("pi5" "murdock")
REMOTE_DIR="~/deli"
EXCLUDES=(
    --exclude=target/
    --exclude=.git/
    --exclude=.worktrees/
    --exclude='mutants.out*'
)

# Allow pushing to specific host(s): ./push.sh pi5
if [ $# -gt 0 ]; then
    HOSTS=("$@")
fi

for host in "${HOSTS[@]}"; do
    echo ">>> Pushing to $host..."
    rsync -az --delete "${EXCLUDES[@]}" ./ "$host:$REMOTE_DIR/"
    echo "    $host done."
done
