"""
Birim Testler (Unit Tests) - Faz 1
DataLoader ve FeatureEngine bileşenlerini sentetik ve gerçek verilerle doğrular.
"""

import unittest
import numpy as np
import pandas as pd

from src.feature_engine import FeatureEngine
from src.config import IndicatorConfig, TargetConfig


class TestFeatureEngine(unittest.TestCase):

    def setUp(self):
        # 100 günlük sentetik OHLCV verisi oluştur
        np.random.seed(42)
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        
        # Basit rastgele yürüyüş (random walk) ile fiyat simülasyonu
        returns = np.random.normal(loc=0.001, scale=0.02, size=n)
        price = 100 * np.exp(np.cumsum(returns))
        
        high = price * (1 + np.abs(np.random.normal(0, 0.01, size=n)))
        low = price * (1 - np.abs(np.random.normal(0, 0.01, size=n)))
        open_p = low + np.random.uniform(0.2, 0.8, size=n) * (high - low)
        close = low + np.random.uniform(0.2, 0.8, size=n) * (high - low)
        volume = np.random.uniform(1000, 50000, size=n)

        self.df = pd.DataFrame(
            {
                "Open": open_p,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            },
            index=dates,
        )

        self.engine = FeatureEngine()

    def test_indicators_generation(self):
        """Tüm teknik indikatörlerin üretildiğini ve NaN değerlerin temizlendiğini doğrula."""
        processed = self.engine.build_features(self.df, include_target=True)
        
        # Beklenen kolonların varlığı
        expected_cols = [
            "EMA_20", "EMA_50", "MACD", "MACD_Signal", "MACD_Hist",
            "RSI", "ATR", "NATR", "BB_High", "BB_Low", "BB_Width",
            "Supertrend", "Supertrend_Direction", "OBV", "Volume_Ratio",
            "Fakeout_High_Risk", "Fakeout_Low_Risk", "Target"
        ]
        for col in expected_cols:
            self.assertIn(col, processed.columns, f"{col} kolonu DataFrame'de bulunamadı!")

        # NaN satır kalmamalı
        self.assertEqual(processed.isna().sum().sum(), 0, "İşlenmiş veride beklenmeyen NaN değerler var!")

    def test_supertrend_values(self):
        """Supertrend yönünün yalnızca 1 veya -1 olduğunu doğrula."""
        processed = self.engine.build_features(self.df, include_target=False)
        directions = processed["Supertrend_Direction"].unique()
        for d in directions:
            self.assertIn(d, [1, -1], f"Geçersiz Supertrend yönü tespit edildi: {d}")

    def test_target_values(self):
        """Hedef değişkenin yalnızca 1 (Al), 0 (Nötr), -1 (Sat) olduğunu doğrula."""
        processed = self.engine.build_features(self.df, include_target=True)
        targets = processed["Target"].unique()
        for t in targets:
            self.assertIn(t, [1, 0, -1], f"Geçersiz hedef sınıfı tespit edildi: {t}")


if __name__ == "__main__":
    unittest.main()
