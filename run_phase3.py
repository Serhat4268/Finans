"""
Faz 3: SQLite Sinyal Loglama, Dinamik Risk Yönetimi ve Backtest Yürütme Scripti
Eğitilen yapay zeka modelinin ürettiği sinyalleri dinamik ATR Stop-Loss / Take-Profit
seviyeleriyle SQLite'a kaydeder, simüle eder ve portföy performans raporu üretir.
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
from src.model_trainer import FinancialModelTrainer
from src.signal_manager import SignalManager
from src.backtester import BacktestEngine


def run_phase3_pipeline(ticker: str = "BTC-USD", model_type: str = "lightgbm"):
    print("=" * 75)
    print(f"[>] FAZ 3 BASLATILIYOR: SQLite Sinyal Loglama & Backtest Motoru ({ticker})")
    print("=" * 75)

    # 1. Adım: Veritabanını Başlat
    signal_mgr = SignalManager()
    print(f"\n[1] SQLite Veritabani Hazirlandi:")
    print(f"    - DB Dosya Yolu: {signal_mgr.db_path}")

    # Önceki test verilerini temizle
    signal_mgr.clear_database(ticker=ticker)

    # 2. Adım: Veri Yükleme ve Model Paketini Çağırma
    print(f"\n[2] Veri ve Egitilmis Model Paketi Yukleniyor...")
    data_loader = DataLoader()
    raw_df = data_loader.fetch_data(ticker=ticker, period="1y", interval="1d", use_cache=True)

    feature_engine = FeatureEngine()
    df_features = feature_engine.build_features(raw_df, include_target=False)
    feature_cols = feature_engine.get_feature_columns(df_features)

    safe_ticker = ticker.replace("-", "_").replace("=", "_").replace("/", "_")
    model_path = Path(__file__).resolve().parent / "models" / f"model_{safe_ticker}_{model_type}.pkl"

    if not model_path.exists():
        print(f"[-] Hata: {model_path} bulunamadi! Lutfen once run_phase2.py calistiriniz.")
        return

    bundle = FinancialModelTrainer.load_model_bundle(model_path)
    model = bundle["model"]
    trained_feats = bundle["feature_names"]
    print(f"    - Yuklenen Model     : {bundle['model_type'].upper()}")
    print(f"    - Oznitelik Uyumu     : {len(trained_feats)} kolon eslesti.")

    # 3. Adım: Tarihsel Sinyal Üretimi ve Dinamik Risk Seviyelerinin Belirlenmesi
    print(f"\n[3] Model Sinyalleri ve Dinamik ATR Risk Seviyeleri Hesaplaniyor...")
    X = df_features[trained_feats]
    preds = model.predict(X)
    probs = model.predict_proba(X)

    signals_records = []
    for i in range(len(df_features)):
        curr_row = df_features.iloc[i]
        timestamp = df_features.index[i]
        pred_idx = preds[i]
        prob_dist = probs[i]

        sig_raw = bundle["target_mapping"].get(pred_idx, 0)
        # Sinyal ters dönüşümü (0 -> -1 SAT, 1 -> 0 NOTR, 2 -> 1 AL)
        if pred_idx == 0:
            sig_raw = -1
            sig_label = "SAT"
        elif pred_idx == 1:
            sig_raw = 0
            sig_label = "NOTR"
        else:
            sig_raw = 1
            sig_label = "AL"

        close = float(curr_row["Close"])
        atr = float(curr_row["ATR"])
        supertrend = float(curr_row["Supertrend"])
        confidence = float(np.max(prob_dist))

        # Dinamik ATR Stop-Loss (1.5x ATR) ve Take-Profit (2.0x Risk:Reward -> 3.0x ATR)
        stop_loss, take_profit = SignalManager.calculate_trade_levels(
            signal_raw=sig_raw,
            close=close,
            atr=atr,
            atr_multiplier=1.5,
            risk_reward_ratio=2.0,
        )

        prob_dict = {
            "SAT": round(float(prob_dist[0]), 4),
            "NOTR": round(float(prob_dist[1]), 4),
            "AL": round(float(prob_dist[2]), 4),
        }

        # SQLite'a kaydet
        sig_id = signal_mgr.log_signal(
            ticker=ticker,
            timestamp=str(timestamp),
            signal_raw=sig_raw,
            signal_label=sig_label,
            confidence=confidence,
            probabilities=prob_dict,
            entry_price=close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            supertrend=supertrend,
            atr=atr,
        )

        signals_records.append({
            "timestamp": timestamp,
            "signal_id": sig_id,
            "signal_raw": sig_raw,
            "signal_label": sig_label,
            "confidence": confidence,
            "entry_price": close,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        })

    df_signals = pd.DataFrame(signals_records).set_index("timestamp")
    print(f"    - SQLite'a Kaydedilen Toplam Sinyal Sayisi: {len(df_signals)}")

    # 4. Adım: Backtest Motorunu Çalıştır
    print(f"\n[4] Backtest Motoru Calistiriliyor (Dinamik Stop-Loss / Take-Profit Simülasyonu)...")
    engine = BacktestEngine(signal_manager=signal_mgr, max_holding_bars=10)
    results = engine.run_backtest(df_bars=df_features, signals_df=df_signals, ticker=ticker, save_to_db=True)

    # SQLite'taki işlemleri çıkış sonuçlarıyla güncelle
    for tr in results["trades"]:
        # Entry time'a göre sinyali bul ve güncelle
        entry_t = tr["entry_time"]
        matched_sig = df_signals[df_signals.index.astype(str) == entry_t]
        if not matched_sig.empty:
            s_id = int(matched_sig.iloc[0]["signal_id"])
            signal_mgr.update_signal_exit(
                signal_id=s_id,
                status=tr["status"],
                exit_price=tr["exit_price"],
                exit_timestamp=tr["exit_time"],
                pnl_pct=tr["pnl_pct"],
            )

    # 5. Adım: Performans Raporunun Gösterimi
    print("\n" + "=" * 75)
    print(f" 📊 PORTFÖY & BACKTEST PERFORMANS RAPORU ({ticker})")
    print("=" * 75)
    print(f"{'Metrik':<30} | {'Deger':<25}")
    print("-" * 75)
    print(f"{'Toplam Islem Sayisi':<30} | {results['total_trades']} adet")
    print(f"{'Basarili / Zararli Islem':<30} | {results['win_trades']} Kazanc / {results['loss_trades']} Kayip")
    print(f"{'Kazanma Orani (Win Rate)':<30} | %{results['win_rate']:.2f}")
    print(f"{'Kar Faktoru (Profit Factor)':<30} | {results['profit_factor']:.2f}")
    print(f"{'Toplam Getiri (Cumulative PnL)':<30} | %{results['total_return_pct']:+.2f}")
    print(f"{'Maksimum Cekilme (Max DD)':<30} | %{results['max_drawdown_pct']:.2f}")
    print(f"{'Yilliklandirilmis Sharpe Orani':<30} | {results['sharpe_ratio']:.2f}")
    print(f"{'Ortalama Islem Getirisi':<30} | %{results['avg_trade_pnl']:+.2f}")
    print("=" * 75)

    # Durum Dağılımı
    print("\n[6] Pozisyon Kapanis Durumlari:")
    status_map = {
        "HIT_TP": "Hedef Kar Alindi (Take-Profit 1:2)",
        "HIT_SL": "Dinamik Stop-Loss Calisti",
        "EXPIRED": "Vade Dolumu (Piyasa Kapanisi)",
    }
    for st, count in results["status_distribution"].items():
        pct = (count / results["total_trades"]) * 100
        print(f"    - {st:<8} [{status_map.get(st, st)}]: {count:2d} adet (%{pct:.1f})")

    # Son 5 İşlemin Detayı
    print("\n[7] Son 5 Islem Detayi (Giris / Cikis / PnL):")
    df_trades = pd.DataFrame(results["trades"])
    display_cols = ["signal", "entry_price", "exit_price", "stop_loss", "take_profit", "status", "pnl_pct", "hold_bars"]
    print(df_trades[display_cols].tail(5).to_string(index=False))

    # SQLite Veritabanı Doğrulaması
    print("\n[8] SQLite Veritabani Dogrulamasi:")
    db_signals = signal_mgr.get_signals(ticker=ticker, limit=3)
    print(f"    - 'signals' tablosunda kayitli son 3 satir:")
    for _, row in db_signals.iterrows():
        print(f"      ID: {row['id']} | {row['timestamp'][:10]} | {row['signal_label']} | Giris: ${row['entry_price']:,.2f} | Durum: {row['status']} | PnL: {row['pnl_pct']}%")

    print("\n" + "=" * 75)
    print("[+] FAZ 3 BASARIYLA TAMAMLANDI: SQLite loglama ve backtest motoru devrede!")
    print("=" * 75)


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    run_phase3_pipeline(ticker=ticker)
