"""Command line interface for AkShare-based A-share data processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ashare_data_processing.providers import build_provider
from ashare_data_processing.quality import quality_report
from ashare_data_processing.storage import CsvDataStore


def _load_config(path: str | Path) -> dict:
    config_path = Path(path)
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch, store, and audit A-share daily panel data."
    )
    parser.add_argument(
        "--config",
        default="config/akshare_example.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--dataset",
        default="market_daily_akshare",
        help="Dataset name used by the CSV data store.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = _load_config(args.config)
    provider = build_provider(config["data"], int(config.get("random_seed", 42)))
    data = provider.fetch(config["data"]["start_date"], config["data"]["end_date"])

    store = CsvDataStore(config["data"].get("storage_dir", "data"))
    store.update(args.dataset, data)
    report = quality_report(data)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
