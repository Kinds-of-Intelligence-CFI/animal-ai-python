import os
import sys
from pathlib import Path


def find_executable(base: Path) -> Path:
    """
    Look for the latest version of the AAI.

    We look in 'BASE/env/' for files matching {AAI,AnimalAI}.{x86_64,exe,app}, e.g. AAI.x86_64.
    Afterward, we look in the folders matching 'BASE/env*/',taking the one that is lexicographically last, e.g. 'BASE/env3.1.3/'.
    """
    error_msg = (
        "Could not automatically find any AAI environment binaries.\n\n"
        "We look in $BASE$/env/ for files matching {AAI,AnimalAI}.{x86_64,exe,app}, e.g. AAI.x86_64. "
        "Afterward, we look in the folders matching '$BASE$/env*/', "
        " taking the one that is lexicographically last, e.g. '$BASE$/env3.1.3/'.\n\n"
        "You can also specify the path exactly with the --env argument."
    ).replace("$BASE$", str(base))

    # Select folder
    env_folders = sorted(base.glob("env*"))
    if (base / "env/").exists():
        env_folders.append(base / "env/")
    if len(env_folders) == 0:
        reason = f"Could not find any folders matching {str(base)}/env*/"
        raise FileNotFoundError(f"{error_msg}\n\nReason: {reason}")
    env_folder = env_folders[-1]

    # Look for binary in selected folder
    # We brace expand manually because glob does not support it.
    binaries = [
        bin
        for bin_name in ["AAI", "AnimalAI", "animalAI", "Animal-AI", "MacOS"]
        for ext in ["x86_64", "exe", "app"]
        for bin in env_folder.glob(f"{bin_name}.{ext}")
    ]
    if len(binaries) == 0:
        reason = f"Could not find any AAI binaries in {env_folder}."
        raise FileNotFoundError(f"{error_msg}\n\nReason: {reason}")

    return binaries[0]


def find_or_download_executable(
    base: Path | None = None, auto_download: bool = True
) -> Path:
    """Find the AAI binary locally, in cache, or by downloading it.

    Search order:
    1. Local directory (if base is provided)
    2. Cached download in ~/.animalai/
    3. Auto-download from GitHub Releases (if enabled)
    """
    from animalai.download import (
        DownloadError,
        download_binary,
        find_cached_executable,
        get_current_platform,
        get_binary_vesion,
        get_release_url,
    )

    # 1. Try local directory
    if base is not None:
        try:
            return find_executable(base)
        except FileNotFoundError:
            pass

    # 2. Try cache
    cached = find_cached_executable()
    if cached is not None:
        return cached

    # 3. Auto-download
    auto_download_env = os.environ.get("ANIMALAI_AUTO_DOWNLOAD", "1").lower()
    if auto_download_env in ("0", "false", "no"):
        auto_download = False

    ci_env = os.environ.get("CI", "").lower()
    if ci_env in ("true", "1"):
        auto_download = False

    if not auto_download:
        raise FileNotFoundError(
            "Could not find any AAI environment binary.\n\n"
            "Options:\n"
            "  1. Download manually from https://github.com/Kinds-of-Intelligence-CFI/animal-ai/releases\n"
            "  2. Run: python -m animalai download\n"
            "  3. Pass the path directly via file_name parameter\n"
            "  4. Place binaries in ./env/ directory"
        )

    # Prompt on TTY
    if sys.stderr.isatty():
        try:
            ver = get_binary_vesion
            platform = get_current_platform()
        except DownloadError as e:
            raise FileNotFoundError(str(e))

        print(
            f"No AAI binary found. About to download v{ver} for {platform} (~400MB).",
            file=sys.stderr,
        )
        try:
            response = input("Continue? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            raise FileNotFoundError("Download cancelled.")
        if response in ("n", "no"):
            raise FileNotFoundError(
                "Download cancelled. Run 'python -m animalai download' later."
            )

    return download_binary()
