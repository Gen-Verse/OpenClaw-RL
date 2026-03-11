#!/usr/bin/env bash
# Pull SWE docker images (SWE-Gym / SWE-bench_Verified) via dockerproxy.net
# and tag them to local canonical registries.
#
# Usage:
#   bash pull_swe_images.sh                  # pull all images in train.jsonl
#   N=10 bash pull_swe_images.sh             # pull first 10 only
#   TRAIN=/path/to/train.jsonl bash pull_swe_images.sh

set -euo pipefail

TRAIN=${TRAIN:-${HOME}/data/swe_gym_subset/train.jsonl}
N=${N:-0}   # 0 = all
# Legacy global overrides (if set, used for all datasets)
PROXY_PREFIX=${PROXY_PREFIX:-}
SRC_PREFIX=${SRC_PREFIX:-}
# Dataset-specific defaults
PROXY_PREFIX_SWE_GYM=${PROXY_PREFIX_SWE_GYM:-dockerproxy.net/xingyaoww}
SRC_PREFIX_SWE_GYM=${SRC_PREFIX_SWE_GYM:-docker.io/xingyaoww}
PROXY_PREFIX_SWE_BENCH=${PROXY_PREFIX_SWE_BENCH:-dockerproxy.net/swebench}
SRC_PREFIX_SWE_BENCH=${SRC_PREFIX_SWE_BENCH:-docker.io/swebench}
MAX_RETRIES=${MAX_RETRIES:-5}
RETRY_SLEEP=${RETRY_SLEEP:-15}
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
LOG_DIR=${LOG_DIR:-${SCRIPT_DIR}/../output/swe_images}

mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/pull_$(date +%F_%H%M%S).log"
exec > >(tee -a "${LOG}") 2>&1

echo "========================================"
echo "start     : $(date)"
echo "train     : ${TRAIN}"
echo "N         : ${N} (0=all)"
echo "proxy(gym): ${PROXY_PREFIX_SWE_GYM}"
echo "tag(gym)  : ${SRC_PREFIX_SWE_GYM}"
echo "proxy(sb) : ${PROXY_PREFIX_SWE_BENCH}"
echo "tag(sb)   : ${SRC_PREFIX_SWE_BENCH}"
if [[ -n "${PROXY_PREFIX}" || -n "${SRC_PREFIX}" ]]; then
  echo "legacy global prefix override enabled"
fi
echo "log       : ${LOG}"
echo "========================================"

python3 - <<PY
import json, subprocess, sys, time
from pathlib import Path

train   = Path("${TRAIN}")
n       = int("${N}")
proxy_legacy = "${PROXY_PREFIX}"
src_legacy   = "${SRC_PREFIX}"
proxy_gym    = "${PROXY_PREFIX_SWE_GYM}"
src_gym      = "${SRC_PREFIX_SWE_GYM}"
proxy_sb     = "${PROXY_PREFIX_SWE_BENCH}"
src_sb       = "${SRC_PREFIX_SWE_BENCH}"
retries = int("${MAX_RETRIES}")
sleep_s = int("${RETRY_SLEEP}")

def resolve_image(record):
    meta = record.get("metadata", {})
    inst = meta.get("instance", {})
    data_source = str(meta.get("data_source", "")).lower()

    iid = inst["instance_id"]
    if "swe-bench" in data_source:
        # Docker doesn't allow double underscore for SWE-bench images.
        iid_compatible = iid.replace("__", "_1776_").lower()
        img_name = f"sweb.eval.x86_64.{iid_compatible}:latest"
        proxy = proxy_sb
        src_pfx = src_sb
    else:
        # SWE-Gym naming convention.
        iid_compatible = iid.replace("__", "_s_").lower()
        img_name = f"sweb.eval.x86_64.{iid_compatible}:latest"
        proxy = proxy_gym
        src_pfx = src_gym

    if proxy_legacy:
        proxy = proxy_legacy
    if src_legacy:
        src_pfx = src_legacy

    return f"{proxy}/{img_name}", f"{src_pfx}/{img_name}"

lines = [l for l in train.open() if l.strip()]
if n > 0:
    lines = lines[:n]
total = len(lines)

failed = []
for i, line in enumerate(lines):
    record = json.loads(line)
    src, dst = resolve_image(record)

    # skip if already tagged
    r = subprocess.run(["docker", "image", "inspect", dst],
                       capture_output=True)
    if r.returncode == 0:
        print(f"[{i+1}/{total}] skip (exists): {dst}", flush=True)
        continue

    # pull with retry
    print(f"[{i+1}/{total}] pull {src}", flush=True)
    ok = False
    for attempt in range(1, retries + 1):
        r = subprocess.run(["docker", "pull", src])
        if r.returncode == 0:
            ok = True
            break
        print(f"  attempt {attempt}/{retries} failed"
              + (f", retry in {sleep_s}s..." if attempt < retries else ""), flush=True)
        if attempt < retries:
            time.sleep(sleep_s)

    if not ok:
        print(f"[{i+1}/{total}] FAILED (skipping): {src}", flush=True)
        failed.append(src)
        continue

    # tag as docker.io/xingyaoww/...
    subprocess.run(["docker", "tag", src, dst], check=True)
    print(f"[{i+1}/{total}] done: {dst}", flush=True)

print()
print(f"finished: {total - len(failed)}/{total} images OK")
if failed:
    print(f"failed ({len(failed)}):")
    for f in failed:
        print(f"  {f}")
    sys.exit(1)
PY

echo "========================================"
echo "end: $(date)"
echo "========================================"
