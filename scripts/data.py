"""CLI for historical market data operations."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from apex_lab.data import (  # noqa: E402
    _resolve_data_dir,
    download_symbol,
    refresh_instruments,
    update_symbol,
)
from apex_lab.data.downloader import VALID_INTERVALS  # noqa: E402
from apex_lab.data.storage import (  # noqa: E402
    get_instruments_path,
    get_metadata_path,
    get_raw_path,
)

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
        choices=sorted(VALID_INTERVALS),
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
        choices=sorted(VALID_INTERVALS),
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


def _configure_logging(debug: bool) -> None:
    """Configure concise CLI logging."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )


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
    """Run the data CLI."""
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
    raise SystemExit(main())
