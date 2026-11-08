"""
Makine Öğrenmesi Modelleme ve Time-Series Split Doğrulama Modülü
XGBoost ve LightGBM kullanarak finansal varlık yön tahmini modellerini zaman serisi
çapraz doğrulamasıyla eğitir, değerlendirir ve canlı sinyal üretimi için serileştirir.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# Hedef etiket eşlemeleri (XGBoost/LightGBM 0..K-1 indeks bekler)
TARGET_TO_ENCODED = {-1: 0, 0: 1, 1: 2}
ENCODED_TO_TARGET = {0: -1, 1: 0, 2: 1}
LABEL_NAMES = {0: "SAT", 1: "NOTR", 2: "AL"}


class FinancialModelTrainer:
    """Zaman serisi uyumlu finansal modelleme, değerlendirme ve tahmin motoru."""

    def __init__(self, n_splits: int = 5, models_dir: Optional[str] = None):
        self.n_splits = n_splits
        if models_dir is None:
            self.models_dir = Path(__file__).resolve().parent.parent / "models"
        else:
            self.models_dir = Path(models_dir).resolve()
            
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def prepare_data(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Öznitelikleri (X) ve kodlanmış hedefi (y) hazırlar.
        Lookahead bias içeren NaN satırları temizler.
        """
        clean_df = df.dropna(subset=feature_cols + ["Target"]).copy()

        X = clean_df[feature_cols]
        # -1, 0, 1 -> 0, 1, 2 dönüşümü
        y = clean_df["Target"].map(TARGET_TO_ENCODED).astype(int)

        return X, y

    def get_model(self, model_type: str = "xgboost", **kwargs) -> Any:
        """Model fabrikası: Varsayılan düzenlileştirilmiş hiperparametrelerle model üretir."""
        if model_type == "xgboost":
            params = {
                "n_estimators": 120,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 2,
                "random_state": 42,
                "eval_metric": "mlogloss",
                "n_jobs": -1,
            }
            params.update(kwargs)
            return XGBClassifier(**params)

        elif model_type == "lightgbm":
            params = {
                "n_estimators": 120,
                "max_depth": 4,
                "learning_rate": 0.04,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 15,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
            params.update(kwargs)
            return LGBMClassifier(**params)

        else:
            raise ValueError(f"Desteklenmeyen model tipi: {model_type}. 'xgboost' veya 'lightgbm' seçiniz.")

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, model_type: str = "xgboost"
    ) -> Dict[str, Any]:
        """
        Geleceğe sızıntıyı (lookahead bias) engelleyen TimeSeriesSplit ile çapraz doğrulama yapar.
        Out-of-Fold (OOF) tahminleri toplayarak genel genelleme yeteneğini ölçer.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)

        fold_metrics = []
        oof_preds = []
        oof_trues = []
        oof_probs = []

        logger.info(f"TimeSeriesSplit ({self.n_splits} fold) baslatiliyor: Model = {model_type.upper()}")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self.get_model(model_type)
            model.fit(X_train, y_train)

            val_preds = model.predict(X_val)
            val_probs = model.predict_proba(X_val)

            acc = accuracy_score(y_val, val_preds)
            f1_macro = f1_score(y_val, val_preds, average="macro", zero_division=0)
            balanced_acc = balanced_accuracy_score(y_val, val_preds)

            fold_metrics.append({
                "fold": fold + 1,
                "train_size": len(train_idx),
                "val_size": len(val_idx),
                "accuracy": acc,
                "f1_macro": f1_macro,
                "balanced_acc": balanced_acc,
            })

            oof_preds.extend(val_preds)
            oof_trues.extend(y_val)
            oof_probs.extend(val_probs)

        # Genel OOF Metrikleri
        oof_trues = np.array(oof_trues)
        oof_preds = np.array(oof_preds)

        overall_acc = accuracy_score(oof_trues, oof_preds)
        overall_f1_macro = f1_score(oof_trues, oof_preds, average="macro", zero_division=0)
        overall_f1_weighted = f1_score(oof_trues, oof_preds, average="weighted", zero_division=0)
        overall_balanced_acc = balanced_accuracy_score(oof_trues, oof_preds)
        conf_mat = confusion_matrix(oof_trues, oof_preds)
        class_rep = classification_report(
            oof_trues,
            oof_preds,
            target_names=[LABEL_NAMES[i] for i in range(3)],
            zero_division=0,
            output_dict=True,
        )

        return {
            "model_type": model_type,
            "fold_metrics": fold_metrics,
            "overall_accuracy": overall_acc,
            "overall_f1_macro": overall_f1_macro,
            "overall_f1_weighted": overall_f1_weighted,
            "overall_balanced_acc": overall_balanced_acc,
            "confusion_matrix": conf_mat,
            "classification_report": class_rep,
        }

    def train_final_model(
        self, X: pd.DataFrame, y: pd.Series, model_type: str = "xgboost"
    ) -> Any:
        """Tüm geçmiş veri üzerinde nihai üretim modelini eğitir."""
        logger.info(f"Nihai {model_type.upper()} modeli egitiliyor (Satir sayisi: {len(X)})...")
        final_model = self.get_model(model_type)
        final_model.fit(X, y)
        return final_model

    def get_feature_importance(
        self, model: Any, feature_names: List[str], top_n: int = 15
    ) -> pd.DataFrame:
        """Eğitilen modelin öznitelik önem düzeylerini çıkarır ve yüzdeye normalize eder."""
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.astype(float)
            total = importances.sum()
            if total > 0:
                importances = importances / total  # Toplamı 1.0 olacak şekilde normalize et
        else:
            importances = np.zeros(len(feature_names))

        df_imp = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        return df_imp.head(top_n)

    def save_model_bundle(
        self,
        model: Any,
        feature_names: List[str],
        ticker: str,
        model_type: str,
        metrics: Dict[str, Any],
        custom_name: Optional[str] = None,
    ) -> Path:
        """
        Modeli, öznitelik sütunlarını ve meta verilerini tek bir pakette (.pkl) saklar.
        Streamlit arayüzü bu paketi okuyarak kusursuz tahmin yapar.
        """
        safe_ticker = ticker.replace("-", "_").replace("=", "_").replace("/", "_")
        filename = custom_name or f"model_{safe_ticker}_{model_type}.pkl"
        save_path = self.models_dir / filename

        # Confusion matrix numpy dizisi olduğu için JSON seri hale getirme desteği
        conf_mat_list = (
            metrics["confusion_matrix"].tolist()
            if isinstance(metrics.get("confusion_matrix"), np.ndarray)
            else metrics.get("confusion_matrix")
        )

        bundle = {
            "model": model,
            "feature_names": feature_names,
            "ticker": ticker,
            "model_type": model_type,
            "target_mapping": TARGET_TO_ENCODED,
            "label_names": LABEL_NAMES,
            "metrics": {
                "accuracy": metrics.get("overall_accuracy"),
                "f1_macro": metrics.get("overall_f1_macro"),
                "f1_weighted": metrics.get("overall_f1_weighted"),
                "balanced_accuracy": metrics.get("overall_balanced_acc"),
                "confusion_matrix": conf_mat_list,
            },
            "created_at": datetime.now().isoformat(),
        }

        joblib.dump(bundle, save_path)
        logger.info(f"Model paketi basariyla kaydedildi: {save_path}")
        return save_path

    @staticmethod
    def load_model_bundle(bundle_path: Path) -> Dict[str, Any]:
        """Kayıtlı model paketini yükler."""
        if not Path(bundle_path).exists():
            raise FileNotFoundError(f"Model dosyasi bulunamadi: {bundle_path}")
        return joblib.load(bundle_path)

    @staticmethod
    def predict_signal(model_bundle: Dict[str, Any], features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Canlı veya test barları için olasılık yüzdeleri ve yön sinyali üretir.

        Döndürür:
            {
                "signal_raw": 1 (Al), 0 (Nötr), -1 (Sat),
                "signal_label": "AL", "NOTR", "SAT",
                "probabilities": {"SAT": 0.12, "NOTR": 0.28, "AL": 0.60},
                "confidence": 0.60
            }
        """
        model = model_bundle["model"]
        feature_names = model_bundle["feature_names"]

        # Öznitelik kolon sıralamasını modele göre garantiye al
        X = features_df[feature_names].tail(1)

        pred_idx = model.predict(X)[0]
        probs = model.predict_proba(X)[0]

        signal_label = LABEL_NAMES[pred_idx]
        signal_raw = ENCODED_TO_TARGET[pred_idx]

        prob_dict = {LABEL_NAMES[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        confidence = round(float(np.max(probs)), 4)

        return {
            "signal_raw": signal_raw,
            "signal_label": signal_label,
            "probabilities": prob_dict,
            "confidence": confidence,
        }
