"""
Faz 1: Veri Çekme, İndikatör Hesaplama ve Sahte Kırılım Filtreleme Doğrulama Scripti
Bu script, veri hattını (data pipeline) ve öznitelik mühendisliği adımlarını uçtan uca test eder.
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# src dizinini path'e ekle
sys.path.append(str(Path(__file__).parent))

from src.data_loader import DataLoader
from src.feature_engine import FeatureEngine
from src.config import DataConfig, IndicatorConfig, TargetConfig


def run_pipeline_demo(ticker: str = "BTC-USD", period: str = "2y", interval: str = "1d", use_cache: bool = True):
    print("=" * 70)
    print(f"[>] FAZ 1 BASLATILIYOR: {ticker} (Periyot: {period}, Aralik: {interval})")
    print("=" * 70)

    # 1. Adım: Veri Çekme
    data_loader = DataLoader()
    raw_df = data_loader.fetch_data(ticker=ticker, period=period, interval=interval, use_cache=use_cache)
    print(f"\n[1] Veri Cekme Tamamlandi:")
    print(f"    - Baslangic Tarihi : {raw_df.index.min().strftime('%Y-%m-%d')}")
    print(f"    - Bitis Tarihi     : {raw_df.index.max().strftime('%Y-%m-%d')}")
    print(f"    - Toplam Bar Sayisi: {len(raw_df)}")
    print(f"    - Kolonlar         : {list(raw_df.columns)}")

    # 2. Adım: Öznitelik Mühendisliği & Sahte Kırılım Filtreleri
    feature_engine = FeatureEngine()
    processed_df = feature_engine.build_features(raw_df, include_target=True)
    features = feature_engine.get_feature_columns(processed_df)

    print(f"\n[2] Oznitelik Muhendisligi Tamamlandi:")
    print(f"    - Uretilen ML Oznitelik Sayisi: {len(features)}")
    print(f"    - Temizlenmis Veri Boyutu     : {processed_df.shape}")
    print(f"    - Ornek Oznitelikler          : {features[:8]} ...")

    # 3. Adım: Sahte Kırılım (Fakeout) Analizi
    bull_fakeouts = int(processed_df["Fakeout_High_Risk"].sum())
    bear_fakeouts = int(processed_df["Fakeout_Low_Risk"].sum())
    print(f"\n[3] Volatilite & Sahte Kirilim Filtresi:")
    print(f"    - Tespit Edilen Boga Tuzagi / Sahte Kirilim (High Risk): {bull_fakeouts} bar")
    print(f"    - Tespit Edilen Ayi Tuzagi / Sahte Cokus (Low Risk)    : {bear_fakeouts} bar")

    # 4. Adım: Faz 2 Hazırlık - Hedef Dağılımı (Target Distribution)
    target_counts = processed_df["Target"].value_counts().sort_index()
    print(f"\n[4] Gelecek Bar Yon Etiketleri Dagilimi (Faz 2 Hazirligi):")
    label_map = {-1: "SAT (Sell)", 0: "NOTR (Hold)", 1: "AL (Buy)"}
    for label, count in target_counts.items():
        pct = (count / len(processed_df)) * 100
        print(f"    - Sinif {label:2d} [{label_map.get(label, 'Bilinmiyor')}]: {count:4d} adet (%{pct:.1f})")

    # 5. Adım: Son 3 Barın Özet Gösterimi
    display_cols = [
        "Close", "RSI", "MACD_Hist", "Supertrend_Direction", "NATR", 
        "Volume_Ratio", "Fakeout_High_Risk", "Target"
    ]
    print("\n[5] Son 3 Bar Veri Incelemesi:")
    print(processed_df[display_cols].tail(3).to_string())

    print("\n" + "=" * 70)
    print("[+] FAZ 1 BASARIYLA TAMAMLANDI: Veri cekme ve indikator motoru hazir!")
    print("=" * 70)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    run_pipeline_demo(ticker=ticker)
