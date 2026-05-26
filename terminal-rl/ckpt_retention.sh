#!/usr/bin/env bash
# Keep only the KEEP_N most recent iter_* checkpoints under SAVE_CKPT.
# Safe to run alongside live training: skips dirs whose mtime is within GRACE_SECS
# (to avoid touching a ckpt while it is being written).
set -euo pipefail

SAVE_CKPT="${SAVE_CKPT:-/nfs/terminal-rl-workspace/OpenClaw-RL/terminal-rl/ckpt/qwen3-4b-terminal-rl}"
KEEP_N="${KEEP_N:-2}"
SLEEP_SECS="${SLEEP_SECS:-300}"   # 5 min
GRACE_SECS="${GRACE_SECS:-180}"   # don't touch dirs modified in last 3 min

log() { echo "[$(date -u +'%F %T')] $*"; }

while true; do
    if [[ -d "$SAVE_CKPT" ]]; then
        now=$(date +%s)
        # candidates: iter_* dirs older than GRACE_SECS, sorted newest→oldest
        mapfile -t iters < <(
            find "$SAVE_CKPT" -maxdepth 1 -type d -name "iter_*" -printf "%T@ %p\n" \
                | sort -rn \
                | awk -v now="$now" -v grace="$GRACE_SECS" '
                    { mtime = $1; $1 = ""; path = substr($0, 2);
                      if (now - mtime > grace) print path
                    }
                '
        )
        total=${#iters[@]}
        if (( total > KEEP_N )); then
            to_delete=( "${iters[@]:$KEEP_N}" )
            log "retention: found $total stable iter_* dirs; keeping $KEEP_N, deleting ${#to_delete[@]}"
            for p in "${to_delete[@]}"; do
                log "  deleting: $(basename "$p")"
                rm -rf "$p"
            done
        fi
    fi
    sleep "$SLEEP_SECS"
done
