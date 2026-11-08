"""
Faz 2: Makine Öğrenmesi Eğitimi, Time-Series Split Doğrulama ve Sinyal Üretim Scripti
XGBoost ve LightGBM modellerini zaman serisi çapraz doğrulamasıyla karşılaştırır,
en iyi modeli serileştirir ve anlık karar destek sinyali simülasyonu yapar.
"""

import io
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Windows konsol UTF-8 desteği
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# src dizinini path'e ekle
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.feature_engine import FeatureEngine
from src.model_trainer import FinancialModelTrainer, LABEL_NAMES


def run_modeling_pipeline(ticker: str = "BTC-USD", period: str = "2y", interval: str = "1d"):
    print("=" * 75)
    print(f"[>] FAZ 2 BASLATILIYOR: Model Egitimi & Zaman Serisi Dogrulama ({ticker})")
    print("=" * 75)

    # 1. Adım: Veri Yükleme ve Öznitelik Mühendisliği
    print("\n[1] Veri ve Oznitelikler Hazirlaniyor...")
    data_loader = DataLoader()
    raw_df = data_loader.fetch_data(ticker=ticker, period=period, interval=interval, use_cache=True)

    feature_engine = FeatureEngine()
    df_features = feature_engine.build_features(raw_df, include_target=True)
    feature_cols = feature_engine.get_feature_columns(df_features)

    print(f"    - Ham Bar Sayisi          : {len(raw_df)}")
    print(f"    - Islenmis Temiz Veri     : {df_features.shape}")
    print(f"    - Kullanilacak Oznitelikler: {len(feature_cols)} adet")

    # 2. Adım: Veri Hazırlığı
    trainer = FinancialModelTrainer(n_splits=5)
    X, y = trainer.prepare_data(df_features, feature_cols)

    # Sınıf Dağılımı
    print("\n[2] Hedef Sinif Dagilimi:")
    counts = y.value_counts().sort_index()
    for enc_label, count in counts.items():
        pct = (count / len(y)) * 100
        print(f"    - Sinif {enc_label} ({LABEL_NAMES[enc_label]:4s}): {count:4d} adet (%{pct:.1f})")

    # 3. Adım: TimeSeriesSplit ile XGBoost ve LightGBM Karşılaştırması
    print("\n[3] Modeller Zaman Serisi Capraz Dogrulamasi (TimeSeriesSplit - 5 Fold) ile Yarisiyor...")
    
    results = {}
    for m_type in ["xgboost", "lightgbm"]:
        print(f"\n    >> {m_type.upper()} Modeli Degerlendiriliyor...")
        res = trainer.cross_validate(X, y, model_type=m_type)
        results[m_type] = res

    # 4. Adım: Model Kıyaslama Tablosu
    print("\n" + "=" * 75)
    print(" 🏆 MODEL KIYASLAMA TABLOSU (Out-of-Fold Performansi)")
    print("=" * 75)
    print(f"{'Metrik':<25} | {'XGBoost':<20} | {'LightGBM':<20}")
    print("-" * 75)
    print(f"{'Dogruluk (Accuracy)':<25} | %{results['xgboost']['overall_accuracy']*100:<19.2f} | %{results['lightgbm']['overall_accuracy']*100:<19.2f}")
    print(f"{'Dengeli Dogruluk (Bal. Acc)':<25} | %{results['xgboost']['overall_balanced_acc']*100:<19.2f} | %{results['lightgbm']['overall_balanced_acc']*100:<19.2f}")
    print(f"{'Macro F1-Score':<25} | {results['xgboost']['overall_f1_macro']:<20.4f} | {results['lightgbm']['overall_f1_macro']:<20.4f}")
    print(f"{'Weighted F1-Score':<25} | {results['xgboost']['overall_f1_weighted']:<20.4f} | {results['lightgbm']['overall_f1_weighted']:<20.4f}")
    print("=" * 75)

    # En iyi modeli seç (Macro F1'e göre)
    best_type = "xgboost" if results["xgboost"]["overall_f1_macro"] >= results["lightgbm"]["overall_f1_macro"] else "lightgbm"
    best_metrics = results[best_type]
    print(f"\n[*] En Basarili Model Secildi: {best_type.upper()} (Macro F1: {best_metrics['overall_f1_macro']:.4f})")

    # 5. Adım: Sınıf Bazında Detaylı Sınıflandırma Raporu & Hata Matrisi
    print(f"\n[5] {best_type.upper()} Sinif Bazli Performans Raporu:")
    rep = best_metrics["classification_report"]
    print(f"    {'Sinif':<10} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Support':<10}")
    print("    " + "-" * 62)
    for cls_name in ["SAT", "NOTR", "AL"]:
        c_stats = rep[cls_name]
        print(f"    {cls_name:<10} | {c_stats['precision']:<12.3f} | {c_stats['recall']:<12.3f} | {c_stats['f1-score']:<12.3f} | {int(c_stats['support']):<10d}")

    print(f"\n    Hata Matrisi (Confusion Matrix - Gercek [Satir] x Tahmin [Sutun]):")
    print(f"    (Sirasiyla: SAT, NOTR, AL)")
    conf_df = pd.DataFrame(
        best_metrics["confusion_matrix"],
        index=[f"Gercek_{k}" for k in ["SAT", "NOTR", "AL"]],
        columns=[f"Tahmin_{k}" for k in ["SAT", "NOTR", "AL"]],
    )
    for line in conf_df.to_string().split("\n"):
        print(f"    {line}")

    # 6. Adım: Nihai Modelin Eğitilmesi ve Kaydedilmesi
    print(f"\n[6] Nihai {best_type.upper()} Modeli Tum Veri Setiyle Egitiliyor...")
    final_model = trainer.train_final_model(X, y, model_type=best_type)

    save_path = trainer.save_model_bundle(
        model=final_model,
        feature_names=feature_cols,
        ticker=ticker,
        model_type=best_type,
        metrics=best_metrics,
    )
    print(f"    - Model Basariyla Serilestirildi (.pkl): {save_path.name}")

    # 7. Adım: Öznitelik Önem Düzeyleri (Feature Importance)
    print(f"\n[7] Karar Mekanizmasini Yoneten En Onemli 10 Gosterge (Feature Importance):")
    imp_df = trainer.get_feature_importance(final_model, feature_cols, top_n=10)
    for i, row in imp_df.iterrows():
        pct = row['Importance'] * 100
        bar = "█" * int(row["Importance"] * 40)
        print(f"    {i+1:2d}. {row['Feature']:<20} : %{pct:5.2f} {bar}")

    # 8. Adım: Canlı Sinyal ve Karar Destek Simülasyonu
    print(f"\n[8] ANLIK KARAR DESTEK VE CANLI SINYAL SIMULASYONU:")
    bundle = FinancialModelTrainer.load_model_bundle(save_path)
    latest_signal = FinancialModelTrainer.predict_signal(bundle, df_features)

    latest_bar = df_features.iloc[-1]
    last_close = latest_bar["Close"]
    last_atr = latest_bar["ATR"]
    last_supertrend = latest_bar["Supertrend"]

    # Dinamik ATR Stop-Loss / Take-Profit Hesaplaması
    atr_multiplier = 1.5
    if latest_signal["signal_raw"] == 1:  # AL
        stop_loss = last_close - (last_atr * atr_multiplier)
        take_profit = last_close + (last_atr * atr_multiplier * 2)
    elif latest_signal["signal_raw"] == -1:  # SAT
        stop_loss = last_close + (last_atr * atr_multiplier)
        take_profit = last_close - (last_atr * atr_multiplier * 2)
    else:  # NÖTR
        stop_loss = np.nan
        take_profit = np.nan

    print("    " + "-" * 55)
    print(f"    Varlik Kodu             : {ticker}")
    print(f"    Son Kapanis Fiyati      : ${last_close:,.2f}")
    print(f"    Uretilen AI Sinyali     : >> {latest_signal['signal_label']} << (Kod: {latest_signal['signal_raw']})")
    print(f"    Model Guven Skoru       : %{latest_signal['confidence']*100:.1f}")
    print(f"    Olasilik Dagilimi       : P(SAT): %{latest_signal['probabilities']['SAT']*100:.1f} | "
          f"P(NOTR): %{latest_signal['probabilities']['NOTR']*100:.1f} | "
          f"P(AL): %{latest_signal['probabilities']['AL']*100:.1f}")
    
    if latest_signal["signal_raw"] != 0:
        print(f"    Dinamik ATR Stop-Loss   : ${stop_loss:,.2f} (-%{abs((stop_loss-last_close)/last_close)*100:.2f})")
        print(f"    Dinamik Hedef (TP 1:2)  : ${take_profit:,.2f} (+%{abs((take_profit-last_close)/last_close)*100:.2f})")
    else:
        print(f"    Pozisyon Durumu         : Piyasa yatay / kararsiz. Nakitte beklenmesi onerilir.")

    print(f"    Supertrend Seviyesi     : ${last_supertrend:,.2f} (Trend: {'YUKSELIS' if latest_bar['Supertrend_Direction']==1 else 'DUSUS'})")
    print("    " + "-" * 55)

    print("\n" + "=" * 75)
    print("[+] FAZ 2 BASARIYLA TAMAMLANDI: Model egitildi, test edildi ve .pkl kaydedildi!")
    print("=" * 75)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    run_modeling_pipeline(ticker=ticker)
