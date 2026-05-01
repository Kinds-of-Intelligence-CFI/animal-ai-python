"""Tests for the animalai.download module."""

import hashlib
import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from animalai.download import (
    ARCHIVE_TEMPLATE,
    DownloadError,
    UnsupportedPlatformError,
    cleanup_old_versions,
    download_binary,
    download_file,
    extract_archive,
    find_cached_executable,
    get_cache_dir,
    get_cache_info,
    get_current_platform,
    get_package_version,
    get_binary_version,
    get_release_url,
    verify_checksum,
)


class TestGetCacheDir(unittest.TestCase):
    def test_default_cache_dir(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ANIMALAI_CACHE", None)
            result = get_cache_dir()
            self.assertEqual(result, Path.home() / ".animalai")

    def test_custom_cache_dir(self):
        with patch.dict(os.environ, {"ANIMALAI_CACHE": "/tmp/custom-cache"}):
            result = get_cache_dir()
            self.assertEqual(result, Path("/tmp/custom-cache"))


class TestGetPackageVersion(unittest.TestCase):
    @patch("animalai.download.version")
    def test_installed_package(self, mock_version):
        mock_version.return_value = "6.0.0"
        self.assertEqual(get_package_version(), "6.0.0")

    @patch("animalai.download.version")
    def test_fallback_to_pyproject(self, mock_version):
        from importlib.metadata import PackageNotFoundError

        mock_version.side_effect = PackageNotFoundError("animalai")
        # Should fall back to reading pyproject.toml from the repo
        # This works in dev because pyproject.toml exists relative to the module
        result = get_package_version()
        self.assertRegex(result, r"\d+\.\d+\.\d+")
    


class TestGetCurrentPlatform(unittest.TestCase):
    @patch("animalai.download.sys")
    def test_linux(self, mock_sys):
        mock_sys.platform = "linux"
        self.assertEqual(get_current_platform(), "Linux")

    @patch("animalai.download.sys")
    def test_windows(self, mock_sys):
        mock_sys.platform = "win32"
        self.assertEqual(get_current_platform(), "Windows")

    @patch("animalai.download.sys")
    def test_macos(self, mock_sys):
        mock_sys.platform = "darwin"
        self.assertEqual(get_current_platform(), "MacOS")

    @patch("animalai.download.sys")
    def test_unsupported(self, mock_sys):
        mock_sys.platform = "freebsd"
        with self.assertRaises(UnsupportedPlatformError):
            get_current_platform()


class TestURLConstruction(unittest.TestCase):
    def test_release_url(self):
        url = get_release_url("v1.1.1", "Linux")
        self.assertEqual(
            url,
            "https://github.com/Kinds-of-Intelligence-CFI/animal-ai/"
            "releases/download/v1.1.1/Linux.zip",
        )


@patch("animalai.download.time.sleep", return_value=None)
class TestDownloadFile(unittest.TestCase):
    @patch("animalai.download.urlopen")
    def test_successful_download(self, mock_urlopen, _mock_sleep):
        data = b"fake binary content" * 100
        mock_response = MagicMock()
        mock_response.read.side_effect = [data, b""]
        mock_response.headers = {"Content-Length": str(len(data))}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "test.zip"
            download_file("https://example.com/test.zip", dest)
            self.assertTrue(dest.exists())
            self.assertEqual(dest.read_bytes(), data)

    @patch("animalai.download.urlopen")
    def test_retry_on_failure(self, mock_urlopen, _mock_sleep):
        from urllib.error import URLError

        data = b"content after retry"
        mock_response = MagicMock()
        mock_response.read.side_effect = [data, b""]
        mock_response.headers = {"Content-Length": str(len(data))}
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        # First call fails, second succeeds
        mock_urlopen.side_effect = [
            URLError("connection reset"),
            mock_response,
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "test.zip"
            download_file("https://example.com/test.zip", dest)
            self.assertTrue(dest.exists())
            self.assertEqual(mock_urlopen.call_count, 2)

    @patch("animalai.download.urlopen")
    def test_all_retries_exhausted(self, mock_urlopen, _mock_sleep):
        from urllib.error import URLError

        mock_urlopen.side_effect = URLError("persistent failure")

        with tempfile.TemporaryDirectory() as tmpdir:
            dest = Path(tmpdir) / "test.zip"
            with self.assertRaises(DownloadError):
                download_file("https://example.com/test.zip", dest)


class TestExtractArchive(unittest.TestCase):
    def _make_zip(self, files: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, content in files.items():
                zf.writestr(name, content)
        return buf.getvalue()

    def test_extract_files(self):
        zip_data = self._make_zip(
            {
                "animalAI.x86_64": b"fake binary",
                "animalAI_Data/config.txt": b"config data",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test.zip"
            archive_path.write_bytes(zip_data)

            dest = Path(tmpdir) / "extracted"
            extract_archive(archive_path, dest)

            self.assertTrue((dest / "animalAI.x86_64").exists())
            self.assertTrue((dest / "animalAI_Data" / "config.txt").exists())
            self.assertEqual(
                (dest / "animalAI.x86_64").read_bytes(), b"fake binary"
            )

    @unittest.skipUnless(sys.platform == "linux", "chmod has no effect on Windows/macOS")
    @patch("animalai.download.sys")
    def test_sets_executable_permissions_on_linux(self, mock_sys):
        mock_sys.platform = "linux"
        zip_data = self._make_zip({"animalAI.x86_64": b"fake binary"})

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "test.zip"
            archive_path.write_bytes(zip_data)

            dest = Path(tmpdir) / "extracted"
            extract_archive(archive_path, dest)

            binary = dest / "animalAI.x86_64"
            mode = binary.stat().st_mode
            self.assertTrue(mode & 0o111, "Binary should be executable")


class TestFindCachedExecutable(unittest.TestCase):
    def test_finds_cached_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            version_dir = cache_dir / "envs" / "6.0.0"
            platform_dir = version_dir / "Linux"
            platform_dir.mkdir(parents=True)
            (version_dir / ".complete").touch()
            (platform_dir / "animalAI.x86_64").write_bytes(b"binary")

            with patch("animalai.download.get_cache_dir", return_value=cache_dir):
                with patch("animalai.download.get_current_platform", return_value="Linux"):
                    result = find_cached_executable("6.0.0")
                    self.assertIsNotNone(result)
                    self.assertEqual(result.name, "animalAI.x86_64")

    def test_returns_none_without_complete_marker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            platform_dir = cache_dir / "envs" / "6.0.0" / "Linux"
            platform_dir.mkdir(parents=True)
            (platform_dir / "animalAI.x86_64").write_bytes(b"binary")
            # No .complete marker

            with patch("animalai.download.get_cache_dir", return_value=cache_dir):
                with patch("animalai.download.get_current_platform", return_value="Linux"):
                    result = find_cached_executable("6.0.0")
                    self.assertIsNone(result)

    def test_returns_none_when_not_cached(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            with patch("animalai.download.get_cache_dir", return_value=cache_dir):
                with patch("animalai.download.get_current_platform", return_value="Linux"):
                    result = find_cached_executable("6.0.0")
                    self.assertIsNone(result)

    def test_finds_windows_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            version_dir = cache_dir / "envs" / "6.0.0"
            platform_dir = version_dir / "Windows"
            platform_dir.mkdir(parents=True)
            (version_dir / ".complete").touch()
            (platform_dir / "Animal-AI.exe").write_bytes(b"binary")

            with patch("animalai.download.get_cache_dir", return_value=cache_dir):
                with patch("animalai.download.get_current_platform", return_value="Windows"):
                    result = find_cached_executable("6.0.0")
                    self.assertIsNotNone(result)
                    self.assertEqual(result.name, "Animal-AI.exe")

    def test_finds_macos_binary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            version_dir = cache_dir / "envs" / "6.0.0"
            platform_dir = version_dir / "MacOS"
            platform_dir.mkdir(parents=True)
            (version_dir / ".complete").touch()
            (platform_dir / "MacOS.app").write_bytes(b"binary")

            with patch("animalai.download.get_cache_dir", return_value=cache_dir):
                with patch("animalai.download.get_current_platform", return_value="MacOS"):
                    result = find_cached_executable("6.0.0")
                    self.assertIsNotNone(result)
                    self.assertEqual(result.name, "MacOS.app")


class TestCleanupOldVersions(unittest.TestCase):
    def test_removes_old_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            envs_dir = cache_dir / "envs"
            (envs_dir / "5.0.0" / "Linux").mkdir(parents=True)
            (envs_dir / get_binary_version() / "Linux").mkdir(parents=True)

            with patch("animalai.download.get_cache_dir", return_value=cache_dir):
                with patch("animalai.download.get_package_version", return_value="6.0.0"):
                    removed = cleanup_old_versions(keep_current=True)

            self.assertEqual(removed, ["5.0.0"])
            self.assertFalse((envs_dir / "5.0.0").exists())
            self.assertTrue((envs_dir / get_binary_version()).exists())

    def test_removes_all_when_requested(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            envs_dir = cache_dir / "envs"
            (envs_dir / "5.0.0" / "Linux").mkdir(parents=True)
            (envs_dir / "6.0.0" / "Linux").mkdir(parents=True)

            with patch("animalai.download.get_cache_dir", return_value=cache_dir):
                removed = cleanup_old_versions(keep_current=False)

            self.assertEqual(sorted(removed), ["5.0.0", "6.0.0"])

    def test_empty_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            with patch("animalai.download.get_cache_dir", return_value=cache_dir):
                removed = cleanup_old_versions()
            self.assertEqual(removed, [])


class TestCacheInfo(unittest.TestCase):
    def test_cache_info_with_versions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            version_dir = cache_dir / "envs" / "6.0.0"
            platform_dir = version_dir / "Linux"
            platform_dir.mkdir(parents=True)
            (version_dir / ".complete").touch()
            (platform_dir / "animalAI.x86_64").write_bytes(b"x" * 1024)

            with patch("animalai.download.get_cache_dir", return_value=cache_dir):
                with patch("animalai.download.get_package_version", return_value="6.0.0"):
                    info = get_cache_info()

            self.assertEqual(info["version"], "6.0.0")
            self.assertEqual(len(info["versions"]), 1)
            self.assertEqual(info["versions"][0]["version"], "6.0.0")
            self.assertTrue(info["versions"][0]["complete"])
            self.assertIn("Linux", info["versions"][0]["platforms"])


class TestVerifyChecksum(unittest.TestCase):
    def test_valid_checksum_passes(self):
        data = b"hello world"
        digest = hashlib.sha256(data).hexdigest()
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            fname = f.name
        try:
            verify_checksum(Path(fname), f"sha256:{digest}")
        finally:
            os.unlink(fname)

    def test_mismatched_checksum_raises(self):
        data = b"hello world"
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(data)
            fname = f.name
        try:
            with self.assertRaises(DownloadError) as ctx:
                verify_checksum(Path(fname), "sha256:0000dead")
            self.assertIn("Checksum mismatch", str(ctx.exception))
        finally:
            os.unlink(fname)

    def test_invalid_format_raises(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"data")
            fname = f.name
        try:
            with self.assertRaises(DownloadError) as ctx:
                verify_checksum(Path(fname), "nocolon")
            self.assertIn("Invalid checksum format", str(ctx.exception))
        finally:
            os.unlink(fname)


class TestLockFile(unittest.TestCase):
    def test_lock_acquired_and_released(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from animalai.download import _download_lock

            version_dir = Path(tmpdir) / "6.0.0"
            version_dir.mkdir()

            lock_path = version_dir / ".download_lock"
            with _download_lock(version_dir):
                self.assertTrue(lock_path.exists())
                self.assertEqual(lock_path.read_text().strip(), str(os.getpid()))

            self.assertFalse(lock_path.exists())

    def test_stale_lock_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            from animalai.download import _download_lock

            version_dir = Path(tmpdir) / "6.0.0"
            version_dir.mkdir()

            lock_path = version_dir / ".download_lock"
            # Write a stale lock with a PID that doesn't exist
            lock_path.write_text("999999999")

            with _download_lock(version_dir):
                self.assertTrue(lock_path.exists())
                self.assertEqual(lock_path.read_text().strip(), str(os.getpid()))


if __name__ == "__main__":
    unittest.main()
