"""
Birim Testler (Unit Tests) - Faz 2: Model Eğitimi ve Tahmin Doğrulama
FinancialModelTrainer sınıfının TimeSeriesSplit çapraz doğrulamasını,
XGBoost & LightGBM modellerini ve sinyal üretim fonksiyonunu test eder.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.model_trainer import (
    FinancialModelTrainer,
    TARGET_TO_ENCODED,
    LABEL_NAMES,
)


class TestFinancialModelTrainer(unittest.TestCase):

    def setUp(self):
        # 120 satırlık sentetik veri seti
        np.random.seed(42)
        n = 120
        self.feature_cols = [f"feat_{i}" for i in range(10)]
        
        data = {col: np.random.randn(n) for col in self.feature_cols}
        # -1, 0, 1 dengeli hedef
        data["Target"] = np.random.choice([-1, 0, 1], size=n)

        self.df = pd.DataFrame(data)
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = FinancialModelTrainer(n_splits=3, models_dir=self.temp_dir)

    def test_prepare_data(self):
        """X ve y boyutlarının doğruluğunu ve hedef etiket kodlamasını kontrol et."""
        X, y = self.trainer.prepare_data(self.df, self.feature_cols)
        self.assertEqual(len(X), len(self.df))
        self.assertEqual(len(y), len(self.df))
        self.assertTrue(set(y.unique()).issubset({0, 1, 2}))

    def test_cross_validation_and_training(self):
        """TimeSeriesSplit çapraz doğrulamasının metrikler ürettiğini doğrula."""
        X, y = self.trainer.prepare_data(self.df, self.feature_cols)
        
        cv_res = self.trainer.cross_validate(X, y, model_type="xgboost")
        self.assertEqual(len(cv_res["fold_metrics"]), 3)
        self.assertIn("overall_accuracy", cv_res)
        self.assertIn("overall_f1_macro", cv_res)
        self.assertGreater(cv_res["overall_accuracy"], 0.0)

    def test_predict_signal_and_bundle(self):
        """Model paketi kaydetme, yükleme ve canlı sinyal üretimini test et."""
        X, y = self.trainer.prepare_data(self.df, self.feature_cols)
        
        cv_res = self.trainer.cross_validate(X, y, model_type="xgboost")
        final_model = self.trainer.train_final_model(X, y, model_type="xgboost")

        saved_path = self.trainer.save_model_bundle(
            model=final_model,
            feature_names=self.feature_cols,
            ticker="TEST-ASSET",
            model_type="xgboost",
            metrics=cv_res,
        )

        self.assertTrue(Path(saved_path).exists())

        # Yükle ve tahmin üret
        bundle = FinancialModelTrainer.load_model_bundle(saved_path)
        signal = FinancialModelTrainer.predict_signal(bundle, self.df)

        self.assertIn("signal_raw", signal)
        self.assertIn("signal_label", signal)
        self.assertIn("probabilities", signal)
        self.assertIn("confidence", signal)

        # Olasılıklar toplamı yaklaşık 1 olmalı
        total_prob = sum(signal["probabilities"].values())
        self.assertAlmostEqual(total_prob, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()
