"""
Birim Testler (Unit Tests) - Faz 3: SQLite Veritabanı, Sinyal Loglama ve Backtest
SignalManager ve BacktestEngine modüllerinin veri tutarlılığını ve risk hesaplamalarını test eder.
"""

import unittest
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from src.signal_manager import SignalManager
from src.backtester import BacktestEngine


class TestSignalManagerAndBacktest(unittest.TestCase):

    def setUp(self):
        # Geçici bir SQLite veritabanı dosyası oluştur
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_signals.db"
        self.mgr = SignalManager(db_path=str(self.db_path))

    def test_calculate_trade_levels(self):
        """Dinamik ATR Stop-Loss ve Take-Profit seviyelerinin mantığını doğrula."""
        close = 100.0
        atr = 2.0
        atr_mult = 1.5
        rr = 2.0

        # AL Sinyali (Uzun Pozisyon)
        sl_buy, tp_buy = SignalManager.calculate_trade_levels(1, close, atr, atr_mult, rr)
        self.assertEqual(sl_buy, 100.0 - (2.0 * 1.5))  # 97.0
        self.assertEqual(tp_buy, 100.0 + (2.0 * 1.5 * 2.0))  # 106.0
        self.assertLess(sl_buy, close)
        self.assertGreater(tp_buy, close)

        # SAT Sinyali (Kısa Pozisyon - spot_only=False durumunda)
        sl_sell, tp_sell = SignalManager.calculate_trade_levels(-1, close, atr, atr_mult, rr, spot_only=False)
        self.assertEqual(sl_sell, 100.0 + (2.0 * 1.5))  # 103.0
        self.assertEqual(tp_sell, 100.0 - (2.0 * 1.5 * 2.0))  # 94.0
        self.assertGreater(sl_sell, close)
        self.assertLess(tp_sell, close)

        # SAT Sinyali (Spot Modunda - Pozisyon Açılmaz)
        sl_spot_sell, tp_spot_sell = SignalManager.calculate_trade_levels(-1, close, atr, atr_mult, rr, spot_only=True)
        self.assertIsNone(sl_spot_sell)
        self.assertIsNone(tp_spot_sell)

        # NÖTR Sinyali (Pozisyon Açılmaz)
        sl_hold, tp_hold = SignalManager.calculate_trade_levels(0, close, atr, atr_mult, rr)
        self.assertIsNone(sl_hold)
        self.assertIsNone(tp_hold)

    def test_log_and_query_signals(self):
        """Sinyal kaydetme, güncelleme ve sorgulama akışını doğrula."""
        probs = {"AL": 0.70, "NOTR": 0.20, "SAT": 0.10}
        sig_id = self.mgr.log_signal(
            ticker="BTC-USD",
            timestamp="2026-10-30 12:00:00",
            signal_raw=1,
            signal_label="AL",
            confidence=0.70,
            probabilities=probs,
            entry_price=80000.0,
            stop_loss=78000.0,
            take_profit=84000.0,
            supertrend=79000.0,
            atr=1000.0,
        )

        self.assertGreater(sig_id, 0)

        # Sinyali sorgula
        df_sigs = self.mgr.get_signals(ticker="BTC-USD")
        self.assertEqual(len(df_sigs), 1)
        self.assertEqual(df_sigs.iloc[0]["signal_label"], "AL")
        self.assertEqual(df_sigs.iloc[0]["status"], "OPEN")

        # Pozisyon çıkışı güncelle
        self.mgr.update_signal_exit(
            signal_id=sig_id,
            status="HIT_TP",
            exit_price=84000.0,
            exit_timestamp="2026-10-31 15:00:00",
            pnl_pct=5.0,
        )

        updated_sig = self.mgr.get_latest_signal(ticker="BTC-USD")
        self.assertEqual(updated_sig["status"], "HIT_TP")
        self.assertEqual(updated_sig["pnl_pct"], 5.0)

    def test_backtest_engine_execution(self):
        """BacktestEngine'in işlemler ve performans metrikleri ürettiğini doğrula."""
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        
        # Sentetik fiyat hareketi
        prices = np.linspace(100, 150, n)
        df_bars = pd.DataFrame({
            "Open": prices,
            "High": prices + 2,
            "Low": prices - 2,
            "Close": prices,
        }, index=dates)

        # Sentetik sinyaller
        signals = [1 if i % 10 == 0 else 0 for i in range(n)]
        signals_df = pd.DataFrame({
            "signal_raw": signals,
            "stop_loss": prices - 3,
            "take_profit": prices + 6,
            "confidence": 0.75,
        }, index=dates)

        engine = BacktestEngine(signal_manager=self.mgr, max_holding_bars=5)
        results = engine.run_backtest(df_bars, signals_df, ticker="TEST-ASSET", save_to_db=False)

        self.assertIn("total_trades", results)
        self.assertIn("win_rate", results)
        self.assertIn("total_return_pct", results)
        self.assertIn("profit_factor", results)
        self.assertGreater(results["total_trades"], 0)

    def test_prevent_duplicate_signals(self):
        """Aynı sembol ve zaman damgasıyla mükerrer kayıt eklenmediğini doğrula."""
        probs = {"AL": 0.8, "NOTR": 0.1, "SAT": 0.1}
        id1 = self.mgr.log_signal(
            ticker="ETH-USD",
            timestamp="2026-11-01 10:00:00",
            signal_raw=1,
            signal_label="AL",
            confidence=0.8,
            probabilities=probs,
            entry_price=3000.0,
            stop_loss=2900.0,
            take_profit=3200.0,
            supertrend=2950.0,
            atr=50.0,
        )
        id2 = self.mgr.log_signal(
            ticker="ETH-USD",
            timestamp="2026-11-01 10:00:00",
            signal_raw=1,
            signal_label="AL",
            confidence=0.8,
            probabilities=probs,
            entry_price=3000.0,
            stop_loss=2900.0,
            take_profit=3200.0,
            supertrend=2950.0,
            atr=50.0,
        )
        self.assertEqual(id1, id2)
        signals = self.mgr.get_signals(ticker="ETH-USD")
        self.assertEqual(len(signals), 1)


if __name__ == "__main__":
    unittest.main()
