"""Download and cache Animal-AI Unity binaries from GitHub Releases."""

import hashlib
import os
import shutil
import stat
import sys
import time
import zipfile
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GITHUB_REPO = "Kinds-of-Intelligence-CFI/animal-ai"
ARCHIVE_TEMPLATE = "{platform}.zip"
CHUNK_SIZE = 65536
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2

BINARY_NAMES = {
    "Linux": "animalAI.x86_64",
    "Windows": "Animal-AI.exe",
    "MacOS": "MacOS.app",
}

MOST_RECENT_VERSION = "v4.3.0"
CHECKSUMS = {
    "Windows": "sha256:be9c7bc1b620530446d69672701a510a1bd4fd55f6844d5becac3b5efd300a17",
    "Linux": "sha256:64bc4f77aa5fe37e413ec5b09b7cbff06a752cace5eb9d65fee37878267d029f",
    "MacOS": "sha256:16f576b9e0ecb5012ee126a8dc3fe818573080e307c5598fe013b1c5d4f0b842",
}

class DownloadError(Exception):
    pass


class UnsupportedPlatformError(DownloadError):
    pass


def get_cache_dir() -> Path:
    return Path(os.environ.get("ANIMALAI_CACHE", Path.home() / ".animalai"))


def get_package_version() -> str:
    try:
        return version("animalai")
    except PackageNotFoundError:
        # Fallback for development installs
        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject.exists():
            for line in pyproject.read_text().splitlines():
                if line.strip().startswith("version"):
                    return line.split("=")[1].strip().strip('"')
        raise DownloadError(
            "Could not determine animalai package version. "
        )

def get_binary_version() -> str:
    return MOST_RECENT_VERSION

def get_current_platform() -> str:
    platform_map = {
        "linux": "Linux",
        "win32": "Windows",
        "darwin": "MacOS",
    }
    platform = platform_map.get(sys.platform)
    if platform is None:
        raise UnsupportedPlatformError(
            f"Unsupported platform: {sys.platform}. "
            f"Supported platforms: {', '.join(platform_map.values())}"
        )
    return platform


def _release_url(version: str, filename: str) -> str:
    return (
        f"https://github.com/{GITHUB_REPO}/releases/download/"
        f"{version}/{filename}"
    )


def get_release_url(ver: str, platform: str) -> str:
    archive = ARCHIVE_TEMPLATE.format(platform=platform)
    return _release_url(ver, archive)


def _progress_bar(current: int, total: int | None, width: int = 40) -> str:
    if total and total > 0:
        fraction = current / total
        filled = int(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        mb_current = current / (1024 * 1024)
        mb_total = total / (1024 * 1024)
        return f"\r  [{bar}] {mb_current:.1f}/{mb_total:.1f} MB ({fraction:.0%})"
    else:
        mb_current = current / (1024 * 1024)
        return f"\r  Downloaded {mb_current:.1f} MB..."


def download_file(url: str, dest: Path, timeout: int = 30) -> None:
    """Download a file with retries and progress display."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_error = None
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            delay = RETRY_BASE_DELAY ** attempt
            print(
                f"  Retry {attempt}/{MAX_RETRIES - 1} after {delay}s..."
            )
            time.sleep(delay)

        try:
            req = Request(url, headers={"User-Agent": "animalai-python"})
            with urlopen(req, timeout=timeout) as response:
                total = response.headers.get("Content-Length")
                total_size = int(total) if total else None

                with open(dest, "wb") as f:
                    downloaded = 0
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if sys.stdout.isatty():
                            print(
                                _progress_bar(downloaded, total_size),
                                end="",
                                flush=True,
                            )

                if sys.stdout.isatty():
                    print(file=sys.stdout)
                return

        except (HTTPError, URLError, TimeoutError, OSError) as e:
            last_error = e
            if dest.exists():
                dest.unlink()

    raise DownloadError(
        f"Failed to download {url} after {MAX_RETRIES} attempts: {last_error}"
    )



def verify_checksum(file_path: Path, expected: str) -> None:
    """Verify a file's checksum against an expected 'algorithm:hex' string."""
    algo, _, expected_digest = expected.partition(":")
    if not expected_digest:
        raise DownloadError(f"Invalid checksum format: {expected}")
    h = hashlib.new(algo)
    with open(file_path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            h.update(chunk)
    actual_digest = h.hexdigest()
    if actual_digest != expected_digest:
        raise DownloadError(
            f"Checksum mismatch for {file_path.name}:\n"
            f"  expected: {expected}\n"
            f"  actual:   {algo}:{actual_digest}"
        )


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a ZIP archive and set executable permissions on Linux/macOS."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(dest_dir)

    if sys.platform != "win32":
        # Set executable permissions on binaries
        for item in dest_dir.rglob("*"):
            if item.is_file() and (
                item.suffix in (".x86_64", ".so", "")
                or item.name.endswith(".app")
                or "MacOS" in str(item)
            ):
                item.chmod(item.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


@contextmanager
def _download_lock(version_dir: Path):
    """Simple PID-based lockfile to prevent concurrent downloads."""
    lock_path = version_dir / ".download_lock"
    version_dir.mkdir(parents=True, exist_ok=True)

    # Check for stale lock
    if lock_path.exists():
        try:
            pid = int(lock_path.read_text().strip())
            os.kill(pid, 0)  # Check if process is alive
            raise DownloadError(
                f"Another download is in progress (PID {pid}). "
                f"If this is wrong, delete {lock_path}"
            )
        except (ValueError, ProcessLookupError, PermissionError):
            lock_path.unlink(missing_ok=True)

    lock_path.write_text(str(os.getpid()))
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def find_cached_executable(ver: str | None = None) -> Path | None:
    """Look for an already-downloaded binary in the cache directory."""
    if ver is None:
        try:
            ver = get_binary_version()
        except DownloadError:
            return None

    platform = get_current_platform()
    cache_dir = get_cache_dir()
    version_dir = cache_dir / "envs" / ver

    # Must have .complete marker
    if not (version_dir / ".complete").exists():
        return None

    platform_dir = version_dir / platform
    if not platform_dir.exists():
        return None

    # Search using same patterns as find_executable
    for bin_name in ["AAI", "AnimalAI", "animalAI", "Animal-AI", "MacOS"]:
        for ext in ["x86_64", "exe", "app"]:
            matches = list(platform_dir.glob(f"{bin_name}.{ext}"))
            if matches:
                return matches[0]

    return None


def download_binary(
    ver: str | None = None,
    platform: str | None = None,
    force: bool = False,
) -> Path:
    """Download the AAI binary for the given version and platform.

    Returns the path to the platform directory containing the extracted binary.
    """
    if ver is None:
        ver = get_binary_version()
    if platform is None:
        platform = get_current_platform()

    cache_dir = get_cache_dir()
    version_dir = cache_dir / "envs" / ver
    platform_dir = version_dir / platform
    complete_marker = version_dir / ".complete"

    # Already downloaded?
    if complete_marker.exists() and platform_dir.exists() and not force:
        cached = find_cached_executable(ver)
        if cached is not None:
            print(f"Binary already cached at {platform_dir}")
            return cached
        # .complete exists but binary not found -- re-download
        print(
            "Cache marker exists but binary not found. Re-downloading..."
        )

    if force and platform_dir.exists():
        shutil.rmtree(platform_dir)
        complete_marker.unlink(missing_ok=True)

    archive_name = ARCHIVE_TEMPLATE.format(platform=platform, version=ver)
    archive_url = get_release_url(ver, platform)
    archive_path = version_dir / archive_name

    with _download_lock(version_dir):
        print(
            f"Downloading {archive_name} from GitHub Releases...",
        )
        try:
            download_file(archive_url, archive_path)
        except DownloadError:
            raise DownloadError(
                f"Failed to download AAI binary.\n"
                f"You can download it manually from:\n"
                f"  {archive_url}\n"
                f"Extract it to: {platform_dir}"
            )

        print("Verifying checksum...")
        verify_checksum(archive_path, CHECKSUMS[platform])

        # Extract
        print(f"Extracting to {version_dir}...")
        extract_archive(archive_path, version_dir)

        # Clean up archive
        archive_path.unlink(missing_ok=True)

        # Mark complete
        complete_marker.touch()

    executable = find_cached_executable(ver)
    if executable is None:
        raise DownloadError(
            f"Extraction succeeded but could not find executable in {platform_dir}. "
            f"Expected one of: {', '.join(BINARY_NAMES.values())}"
        )

    print(f"Done. Binary at: {executable}")
    return executable


def cleanup_old_versions(keep_current: bool = True) -> list[str]:
    """Remove old cached versions. Returns list of removed version strings."""
    cache_dir = get_cache_dir()
    envs_dir = cache_dir / "envs"
    if not envs_dir.exists():
        return []

    try:
        current_version = get_binary_version() if keep_current else None
    except DownloadError:
        current_version = None

    removed = []
    for version_dir in envs_dir.iterdir():
        if not version_dir.is_dir():
            continue
        if keep_current and version_dir.name == current_version:
            continue
        shutil.rmtree(version_dir)
        removed.append(version_dir.name)

    return removed


def get_cache_info() -> dict:
    """Return information about the cache for display purposes."""
    cache_dir = get_cache_dir()
    envs_dir = cache_dir / "envs"

    try:
        pkg_version = get_package_version()
    except DownloadError:
        pkg_version = "unknown"

    info = {
        "version": pkg_version,
        "cache_dir": str(cache_dir),
        "versions": [],
    }

    if envs_dir.exists():
        for version_dir in sorted(envs_dir.iterdir()):
            if not version_dir.is_dir():
                continue
            complete = (version_dir / ".complete").exists()
            platforms = [
                p.name
                for p in version_dir.iterdir()
                if p.is_dir() and p.name in ("Linux", "Windows", "MacOS")
            ]
            # Calculate size
            total_size = sum(
                f.stat().st_size
                for f in version_dir.rglob("*")
                if f.is_file()
            )
            info["versions"].append(
                {
                    "version": version_dir.name,
                    "complete": complete,
                    "platforms": platforms,
                    "size_mb": round(total_size / (1024 * 1024), 1),
                }
            )

    return info
