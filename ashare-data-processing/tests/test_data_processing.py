from __future__ import annotations

import tempfile
import unittest

import pandas as pd

from ashare_data_processing.providers import AkShareProvider, SyntheticProvider
from ashare_data_processing.quality import quality_report
from ashare_data_processing.storage import CsvDataStore


class DataProcessingTest(unittest.TestCase):
    def test_synthetic_quality_report(self) -> None:
        data = SyntheticProvider(symbol_count=5, random_seed=1).fetch(
            "2024-01-01", "2024-12-31"
        )
        report = quality_report(data)
        self.assertEqual(report["symbol_count"], 5)
        self.assertEqual(report["duplicate_key_count"], 0)
        self.assertIn("core_missing_ratios", report)

    def test_csv_store_incremental_update(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = CsvDataStore(temp_dir)
            first = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02"]),
                    "symbol": ["000001.SZ"],
                    "close": [10.0],
                }
            )
            second = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                    "symbol": ["000001.SZ", "000001.SZ"],
                    "close": [10.1, 10.2],
                }
            )
            store.update("market_daily_akshare", first)
            store.update("market_daily_akshare", second)
            saved = store.load("market_daily_akshare", parse_dates=["date"])
            self.assertEqual(len(saved), 2)
            self.assertEqual(float(saved.iloc[0]["close"]), 10.1)

    def test_calendar_completion_does_not_create_pre_listing_rows(self) -> None:
        provider = AkShareProvider(symbols=["000001.SZ"], max_symbols=1)
        raw = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]
                ),
                "symbol": ["A", "A", "B", "B"],
                "close": [10.0, 10.1, 20.0, 20.2],
                "open": [10.0, 10.1, 20.0, 20.2],
                "high": [10.0, 10.1, 20.0, 20.2],
                "low": [10.0, 10.1, 20.0, 20.2],
                "volume": [1, 1, 1, 1],
                "amount": [10, 10, 20, 20],
                "turnover_rate": [0.01] * 4,
                "outstanding_share": [100] * 4,
            }
        )
        completed = provider._complete_calendar(raw)
        self.assertEqual(
            completed.loc[completed["symbol"] == "B", "date"].min(),
            pd.Timestamp("2024-01-02"),
        )


if __name__ == "__main__":
    unittest.main()
