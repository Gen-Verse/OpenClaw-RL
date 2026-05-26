"""Apply Ray patches for 1-GPU terminal-RL demo.

On this 8xH200 box we don't strictly need the GPU-count hack (Actor+Rollout+PRM
fit on 3 separate GPUs natively), but the script is kept idempotent so it's safe
to re-run on any fresh env. It prints WARN if the patterns don't match the
installed Ray version and skips rather than corrupting the files.
"""
import ast, site, os, sys

def _find_sp():
    for sp in site.getsitepackages():
        if os.path.exists(os.path.join(sp, "ray")):
            return sp
    for p in sys.path:
        if os.path.exists(os.path.join(p, "ray")):
            return p
    return site.getsitepackages()[0]

SP = _find_sp()

# Patch 1: allow --num-gpus to exceed CUDA_VISIBLE_DEVICES count
p1 = f"{SP}/ray/_private/resource_and_label_spec.py"
with open(p1) as f:
    src = f.read()
old1 = (
    "        if (\n"
    "            num_accelerators is not None\n"
    "            and visible_accelerator_ids is not None\n"
    "            and num_accelerators > len(visible_accelerator_ids)\n"
    "        ):\n"
    "            raise ValueError(\n"
    "                f\"Attempting to start raylet with {num_accelerators} \"\n"
    "                f\"{accelerator_resource_name}, \"\n"
    "                f\"but {accelerator_manager.get_visible_accelerator_ids_env_var()} \"\n"
    "                f\"contains {visible_accelerator_ids}.\"\n"
    "            )"
)
new1 = (
    "        if (\n"
    "            num_accelerators is not None\n"
    "            and visible_accelerator_ids is not None\n"
    "            and num_accelerators > len(visible_accelerator_ids)\n"
    "        ):\n"
    "            pass  # patched: allow num_gpus > visible GPU count (1-GPU hack)"
)
if "patched: allow num_gpus" not in src:
    if old1 in src:
        src = src.replace(old1, new1)
        open(p1, "w").write(src)
        print("[patch1] resource_and_label_spec.py patched")
    else:
        print("[patch1] WARN: pattern not found, Ray version may differ")
else:
    print("[patch1] already patched")
ast.parse(open(p1).read())
print("[patch1] syntax OK")

# Patch 2: clamp GPU index so all tasks land on GPU 0
p2 = f"{SP}/ray/_private/worker.py"
with open(p2) as f:
    src2 = f.read()
old2 = "            assigned_ids = {str(original_ids[i]) for i in assigned_ids}"
new2 = (
    "            # patched: clamp index so all tasks land on GPU 0\n"
    "            assigned_ids = {str(original_ids[i % len(original_ids)]) for i in assigned_ids}"
)
if "patched: clamp index" not in src2:
    if old2 in src2:
        src2 = src2.replace(old2, new2, 1)
        open(p2, "w").write(src2)
        print("[patch2] worker.py patched")
    else:
        print("[patch2] WARN: pattern not found, Ray version may differ")
else:
    print("[patch2] already patched")
ast.parse(open(p2).read())
print("[patch2] syntax OK")
print("All Ray patches applied.")
