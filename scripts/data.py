"""CLI for historical market data operations."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

logger = logging.getLogger(__name__)

# The public download_symbol() API does not accept an exchange override, so the
# CLI only exposes the default NSE lookup path required by this PR.
SUPPORTED_EXCHANGES: tuple[str, ...] = ("NSE",)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    shared_parser = argparse.ArgumentParser(add_help=False)
    shared_parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full Python tracebacks on failures.",
    )

    parser = argparse.ArgumentParser(
        description="Historical data CLI for the APEX Lab data engine.",
        parents=[shared_parser],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser(
        "download",
        description="Download full historical data for a symbol.",
        help="Download full historical data for a symbol.",
        parents=[shared_parser],
    )
    download_parser.add_argument("--symbol", required=True, help="Trading symbol to download.")
    download_parser.add_argument(
        "--interval",
        required=True,
        choices=_valid_intervals(),
        help="Candle interval to download.",
    )
    download_parser.add_argument(
        "--from",
        dest="from_date",
        required=True,
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    download_parser.add_argument(
        "--to",
        dest="to_date",
        default="today",
        help="Inclusive end date in YYYY-MM-DD format or 'today'.",
    )
    download_parser.add_argument(
        "--exchange",
        default="NSE",
        choices=SUPPORTED_EXCHANGES,
        help="Exchange to use for the symbol lookup. Only NSE is exposed by this CLI.",
    )
    download_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard any cached download artifacts before downloading.",
    )

    update_parser = subparsers.add_parser(
        "update",
        description="Download only missing data for a symbol.",
        help="Download only missing data for a symbol.",
        parents=[shared_parser],
    )
    update_parser.add_argument("--symbol", required=True, help="Trading symbol to update.")
    update_parser.add_argument(
        "--interval",
        required=True,
        choices=_valid_intervals(),
        help="Candle interval to update.",
    )

    subparsers.add_parser(
        "refresh-instruments",
        description="Refresh the instrument master file.",
        help="Refresh the instrument master file.",
        parents=[shared_parser],
    )

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    return build_parser().parse_args(argv)


def _ensure_src_root_on_path() -> None:
    """Add the repository src directory to sys.path when running as a script."""
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))


def _valid_intervals() -> list[str]:
    """Return supported download intervals from the existing data engine."""
    _ensure_src_root_on_path()

    from apex_lab.data.downloader import VALID_INTERVALS  # noqa: PLC0415

    return sorted(VALID_INTERVALS)


def _resolve_data_dir() -> Path:
    """Resolve the active data directory using the existing data API."""
    _ensure_src_root_on_path()

    from apex_lab.data import _resolve_data_dir as resolve_data_dir  # noqa: PLC0415

    return resolve_data_dir()


def download_symbol(symbol: str, interval: str, start_date: str, end_date: str) -> Any:
    """Dispatch to the existing download_symbol API."""
    _ensure_src_root_on_path()

    from apex_lab.data import download_symbol as data_download_symbol  # noqa: PLC0415

    return data_download_symbol(symbol, interval, start_date, end_date)


def update_symbol(symbol: str, interval: str) -> Any:
    """Dispatch to the existing update_symbol API."""
    _ensure_src_root_on_path()

    from apex_lab.data import update_symbol as data_update_symbol  # noqa: PLC0415

    return data_update_symbol(symbol, interval)


def refresh_instruments() -> Any:
    """Dispatch to the existing refresh_instruments API."""
    _ensure_src_root_on_path()

    from apex_lab.data import refresh_instruments as data_refresh_instruments  # noqa: PLC0415

    return data_refresh_instruments()


def get_raw_path(data_dir: Path, interval: str, symbol: str) -> Path:
    """Return the existing raw-data output path for a symbol."""
    _ensure_src_root_on_path()

    from apex_lab.data.storage import get_raw_path as storage_get_raw_path  # noqa: PLC0415

    return storage_get_raw_path(data_dir, interval, symbol)


def get_metadata_path(data_dir: Path, interval: str, symbol: str) -> Path:
    """Return the existing metadata output path for a symbol."""
    _ensure_src_root_on_path()

    from apex_lab.data.storage import (
        get_metadata_path as storage_get_metadata_path,  # noqa: PLC0415
    )

    return storage_get_metadata_path(data_dir, interval, symbol)


def get_instruments_path(data_dir: Path) -> Path:
    """Return the existing instruments output path."""
    _ensure_src_root_on_path()

    from apex_lab.data.storage import (
        get_instruments_path as storage_get_instruments_path,  # noqa: PLC0415
    )

    return storage_get_instruments_path(data_dir)


def _configure_logging(debug: bool) -> None:
    """Configure concise CLI logging."""
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False


def _clear_download_artifacts(data_dir: Path, symbol: str, interval: str) -> None:
    """Remove existing files and caches for a fresh download."""
    raw_path = get_raw_path(data_dir, interval, symbol)
    metadata_path = get_metadata_path(data_dir, interval, symbol)
    cache_dir = data_dir / "cache" / f"{symbol}_{interval}"

    if raw_path.exists():
        raw_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _run_download(args: argparse.Namespace) -> int:
    """Execute the download command."""
    data_dir = _resolve_data_dir()

    logger.info("Downloading %s", args.symbol)
    logger.info("Interval: %s", args.interval)
    logger.info("From: %s", args.from_date)
    logger.info("To: %s", args.to_date)
    logger.info("Exchange: %s", args.exchange)

    if args.overwrite:
        logger.info("Overwrite: enabled")
        _clear_download_artifacts(data_dir, args.symbol, args.interval)

    df = download_symbol(args.symbol, args.interval, args.from_date, args.to_date)
    logger.info("Downloaded %d candles", df.height)
    logger.info("Saved to %s", get_raw_path(data_dir, args.interval, args.symbol))
    return 0


def _run_update(args: argparse.Namespace) -> int:
    """Execute the update command."""
    data_dir = _resolve_data_dir()

    logger.info("Updating %s", args.symbol)
    logger.info("Interval: %s", args.interval)

    df = update_symbol(args.symbol, args.interval)
    logger.info("Dataset now contains %d candles", df.height)
    logger.info("Saved to %s", get_raw_path(data_dir, args.interval, args.symbol))
    return 0


def _run_refresh_instruments(_: argparse.Namespace) -> int:
    """Execute the refresh-instruments command."""
    data_dir = _resolve_data_dir()

    logger.info("Refreshing instrument master")
    df = refresh_instruments()
    logger.info("Refreshed %d instruments", df.height)
    logger.info("Saved to %s", get_instruments_path(data_dir))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the data CLI.

    Args:
        argv: Optional CLI arguments. Defaults to ``sys.argv`` when ``None``.

    Returns:
        Process exit code: ``0`` on success, ``1`` on handled runtime failure.
    """
    args = parse_args(argv)
    _configure_logging(args.debug)

    try:
        handlers = {
            "download": _run_download,
            "update": _run_update,
            "refresh-instruments": _run_refresh_instruments,
        }
        return handlers[args.command](args)
    except Exception as exc:
        if args.debug:
            raise
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
