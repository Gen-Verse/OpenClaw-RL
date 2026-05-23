# Script to download and prepare currently supported datasets for terminal agent training

import os
import shutil
import subprocess
from pathlib import Path

DATASET_DIR = Path(os.getenv("DATASET_DIR", "./terminal-rl/dataset"))


def _git_sparse_checkout(
    repo_url: str, temp_dir: Path, branch: str, sparse_path: str
) -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            repo_url,
            str(temp_dir),
            "-b",
            branch,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(temp_dir), "sparse-checkout", "set", sparse_path],
        check=True,
    )


def _download_github_folder(
    repo_url, sparse_path, target_dir, branch="main", temp_suffix="temp"
):
    """
    General function to download a specific folder from a GitHub repository.

    Args:
        repo_url: GitHub repository URL (.git)
        sparse_path: Path within the repo to download
        target_dir: Local destination directory
        branch: Git branch to checkout (default: "main")
        temp_suffix: Suffix for temporary directory name
    """
    if target_dir.exists():
        print(f"Dataset already exists at {target_dir}. Skipping download.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = DATASET_DIR / f"temp_{temp_suffix}"

    try:
        _git_sparse_checkout(repo_url, temp_dir, branch, sparse_path)

        # Move downloaded folder to target location
        shutil.move(str(temp_dir / sparse_path), str(target_dir))
        print(f"Successfully downloaded to {target_dir}")
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def _download_github_folder_candidates(
    repo_url, sparse_paths, target_dir, branch="main", temp_suffix="temp"
):
    """
    Try several sparse checkout paths and keep the first one that exists.

    This helps us remain compatible with upstream repo reorganizations
    (for example, terminal-bench main moved its task directory from
    `tasks/` to `original-tasks/`).
    """
    if target_dir.exists():
        print(f"Dataset already exists at {target_dir}. Skipping download.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    temp_dir = DATASET_DIR / f"temp_{temp_suffix}"
    last_error = None

    try:
        for sparse_path in sparse_paths:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

            try:
                _git_sparse_checkout(repo_url, temp_dir, branch, sparse_path)
            except subprocess.CalledProcessError as exc:
                last_error = exc
                continue

            source_path = temp_dir / sparse_path
            if source_path.exists():
                shutil.move(str(source_path), str(target_dir))
                print(f"Successfully downloaded to {target_dir} from {sparse_path}")
                return

        tried = ", ".join(sparse_paths)
        raise FileNotFoundError(
            f"Could not find any of the requested paths in {repo_url}@{branch}: {tried}"
        ) from last_error
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def download_seta_env():
    url = "https://github.com/camel-ai/seta-env.git"
    target_dir = DATASET_DIR / "seta_env"
    _download_github_folder(
        url, "Dataset", target_dir, branch="main", temp_suffix="seta_env"
    )


def download_tbench_core():
    url = "https://github.com/laude-institute/terminal-bench.git"
    target_dir = DATASET_DIR / "tbench_core"
    _download_github_folder_candidates(
        url,
        ["original-tasks", "tasks"],
        target_dir,
        branch="main",
        temp_suffix="tbench_core",
    )


def download_tbench_test():
    url = "https://github.com/laude-institute/terminal-bench.git"
    target_dir = DATASET_DIR / "tbench_test"
    _download_github_folder(
        url,
        "tasks",
        target_dir,
        branch="dataset/terminal-bench-core/v0.1.x",
        temp_suffix="tbench_test",
    )


def download_tbench_adapted():
    url = "https://github.com/laude-institute/terminal-bench-datasets.git"
    raw_dir = DATASET_DIR / "tbench_adapted_raw"
    target_dir = DATASET_DIR / "tbench_adapted"

    if target_dir.exists():
        print(f"Dataset already exists at {target_dir}. Skipping download.")
        return

    # Download the raw datasets
    _download_github_folder(
        url, "datasets", raw_dir, branch="main", temp_suffix="tbench_adapted"
    )

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Create symbolic links with prefixed names
    for subfolder in raw_dir.iterdir():
        if subfolder.is_dir():
            subfolder_name = subfolder.name
            for task_folder in subfolder.iterdir():
                if task_folder.is_dir():
                    task_name = task_folder.name
                    prefixed_name = f"{subfolder_name}_{task_name}"
                    symlink_path = target_dir / prefixed_name
                    symlink_path.symlink_to(task_folder, target_is_directory=True)

    print(f"Successfully created symlinks in {target_dir}")


def download_data(ds_name):
    DATASET_DOWNLOADERS = {
        "seta_env": download_seta_env,
        "tbench_core": download_tbench_core,
        "tbench_test": download_tbench_test,
        "tbench_adapted": download_tbench_adapted,
    }
    if ds_name not in DATASET_DOWNLOADERS:
        raise ValueError(f"Dataset {ds_name} is not supported.")
    DATASET_DOWNLOADERS[ds_name]()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        download_data(sys.argv[1])
    else:
        print("Available datasets: seta_env, tbench_core, tbench_test, tbench_adapted")
        print("Usage: python download_data.py <dataset_name>")
