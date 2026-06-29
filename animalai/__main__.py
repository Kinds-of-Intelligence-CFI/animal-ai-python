"""CLI entry point for animalai: python -m animalai <command>"""

import argparse
import sys

from animalai.download import (
    DownloadError,
    cleanup_old_versions,
    download_binary,
    get_cache_info,
)


def cmd_download(args):
    try:
        path = download_binary(
            ver=args.version, platform=args.platform, force=args.force
        )
        print(f"Binary ready at: {path}")
    except DownloadError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_info(args):
    info = get_cache_info()
    print(f"Animal-AI v{info['version']}")
    print(f"Cache directory: {info['cache_dir']}")

    if not info["versions"]:
        print("No cached binaries.")
    else:
        print("Cached versions:")
        for v in info["versions"]:
            status = "complete" if v["complete"] else "incomplete"
            platforms = ", ".join(v["platforms"]) if v["platforms"] else "none"
            print(
                f"  {v['version']} [{status}] - {platforms} ({v['size_mb']:.1f} MB)"
            )
    return 0


def cmd_cleanup(args):
    removed = cleanup_old_versions(keep_current=not args.all)
    if removed:
        for v in removed:
            print(f"Removed v{v}")
        print(f"Cleaned up {len(removed)} version(s).")
    else:
        print("Nothing to clean up.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="animalai", description="Animal-AI command-line tools"
    )
    subparsers = parser.add_subparsers(dest="command")

    dl_parser = subparsers.add_parser(
        "download", help="Download the AAI Unity binary"
    )
    dl_parser.add_argument(
        "--version",
        default=None,
        help="Version to download (default: current package version)",
    )
    dl_parser.add_argument(
        "--platform",
        default=None,
        choices=["Linux", "Windows", "MacOS"],
        help="Platform to download for (default: auto-detect)",
    )
    dl_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if already cached",
    )

    subparsers.add_parser("info", help="Show cache and version info")

    clean_parser = subparsers.add_parser(
        "cleanup", help="Remove old cached versions"
    )
    clean_parser.add_argument(
        "--all",
        action="store_true",
        help="Remove all cached versions including current",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    handlers = {
        "download": cmd_download,
        "info": cmd_info,
        "cleanup": cmd_cleanup,
    }

    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
