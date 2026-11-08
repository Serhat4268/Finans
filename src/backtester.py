"""
Geriye Dönük Test ve Performans İzleme Motoru (Backtest Engine)
Yapay zeka modellerinin ürettiği sinyalleri dinamik ATR Stop-Loss ve Take-Profit
kurallarıyla simüle eder; Win Rate, Sharpe Oranı, Maksimum Çekilme (MDD) ve Kâr Faktörünü hesaplar.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.signal_manager import SignalManager

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Yapay zeka sinyallerinin tarihsel performansını test eden ve metriklerini çıkaran motor."""

    def __init__(
        self,
        signal_manager: Optional[SignalManager] = None,
        max_holding_bars: int = 10,
        risk_free_rate: float = 0.05,
    ):
        self.signal_mgr = signal_manager or SignalManager()
        self.max_holding_bars = max_holding_bars
        self.risk_free_rate = risk_free_rate

    def run_backtest(
        self,
        df_bars: pd.DataFrame,
        signals_df: pd.DataFrame,
        ticker: str = "BTC-USD",
        save_to_db: bool = True,
    ) -> Dict[str, Any]:
        """
        Geçmiş sinyalleri çubuk çubuk (bar-by-bar) simüle ederek gerçeğe en yakın sonuçları üretir.

        Parametreler:
            df_bars: İlgili varlığın 'Open', 'High', 'Low', 'Close' kolonlarını içeren DataFrame.
            signals_df: Her bar için 'signal_raw', 'stop_loss', 'take_profit', 'confidence' içeren DataFrame.
            ticker: Varlık sembolü.
            save_to_db: Çıkan sonuçların SQLite'a kaydedilmesi.

        Döndürür:
            Detaylı performans metrikleri ve işlem geçmişi tablosu.
        """
        trades: List[Dict[str, Any]] = []
        equity = 100.0  # Başlangıç sermayesi (normalize 100 birim)
        equity_curve = [equity]

        # Tarihe göre sıralı olduğundan emin ol
        df = df_bars.copy().sort_index()
        sig_df = signals_df.copy().sort_index()

        n = len(df)
        bar_indices = list(range(n))

        # Aktif pozisyon çakışmalarını önlemek için (tekil pozisyon modeli)
        i = 0
        while i < n - 1:
            timestamp = df.index[i]
            if timestamp not in sig_df.index:
                i += 1
                continue

            sig_row = sig_df.loc[timestamp]
            sig_raw = int(sig_row["signal_raw"]) if "signal_raw" in sig_row else int(sig_row["signal"])
            confidence = float(sig_row.get("confidence", 0.0))

            # Sadece AL (1) veya SAT (-1) sinyallerinde işlem açılır
            if sig_raw in [1, -1]:
                entry_price = float(df["Close"].iloc[i])
                stop_loss = float(sig_row["stop_loss"])
                take_profit = float(sig_row["take_profit"])

                exit_price = None
                exit_time = None
                status = "EXPIRED"
                hold_bars = 0

                # Pozisyon açıldıktan sonraki barlarda fiyat hareketini izle
                for j in range(i + 1, min(i + 1 + self.max_holding_bars, n)):
                    curr_bar = df.iloc[j]
                    curr_time = df.index[j]
                    high_p = curr_bar["High"]
                    low_p = curr_bar["Low"]
                    close_p = curr_bar["Close"]
                    hold_bars += 1

                    if sig_raw == 1:  # UZUN (AL) POZİSYON
                        # 1. Stop-Loss kontrolü
                        if low_p <= stop_loss:
                            exit_price = stop_loss
                            exit_time = curr_time
                            status = "HIT_SL"
                            break
                        # 2. Take-Profit kontrolü
                        elif high_p >= take_profit:
                            exit_price = take_profit
                            exit_time = curr_time
                            status = "HIT_TP"
                            break

                    elif sig_raw == -1:  # KISA (SAT) POZİSYON
                        # 1. Stop-Loss kontrolü (fiyat yukarı kırarsa stop)
                        if high_p >= stop_loss:
                            exit_price = stop_loss
                            exit_time = curr_time
                            status = "HIT_SL"
                            break
                        # 2. Take-Profit kontrolü (fiyat aşağı düşerse hedef kâr)
                        elif low_p <= take_profit:
                            exit_price = take_profit
                            exit_time = curr_time
                            status = "HIT_TP"
                            break

                # Eğer maksimum bekleme süresinde ne SL ne TP vurulduysa, son barın kapanışında çık
                if exit_price is None:
                    last_idx = min(i + self.max_holding_bars, n - 1)
                    exit_price = float(df["Close"].iloc[last_idx])
                    exit_time = df.index[last_idx]
                    status = "EXPIRED"

                # Kâr / Zarar Yüzdesi Hesabı
                if sig_raw == 1:
                    pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                else:
                    pnl_pct = ((entry_price - exit_price) / entry_price) * 100

                # Sermaye büyümesini güncelle
                equity = equity * (1 + (pnl_pct / 100))
                equity_curve.append(equity)

                trade_record = {
                    "ticker": ticker,
                    "entry_time": str(timestamp),
                    "exit_time": str(exit_time),
                    "signal": "AL" if sig_raw == 1 else "SAT",
                    "confidence": round(confidence, 3),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(exit_price, 2),
                    "stop_loss": round(stop_loss, 2),
                    "take_profit": round(take_profit, 2),
                    "status": status,
                    "pnl_pct": round(pnl_pct, 2),
                    "hold_bars": hold_bars,
                }
                trades.append(trade_record)

                # Pozisyon kapandıktan sonraki bara atla
                i += hold_bars
            else:
                i += 1

        # Metriklerin Hesaplanması
        metrics = self._calculate_metrics(trades, equity_curve, ticker, df.index[0], df.index[-1])

        # Veritabanına kaydet (isteğe bağlı)
        if save_to_db and trades:
            self.signal_mgr.log_backtest_run(
                ticker=ticker,
                start_date=str(df.index[0].strftime("%Y-%m-%d")),
                end_date=str(df.index[-1].strftime("%Y-%m-%d")),
                total_trades=metrics["total_trades"],
                win_rate=metrics["win_rate"],
                profit_factor=metrics["profit_factor"],
                total_return_pct=metrics["total_return_pct"],
                max_drawdown_pct=metrics["max_drawdown_pct"],
                sharpe_ratio=metrics["sharpe_ratio"],
            )

        metrics["trades"] = trades
        metrics["equity_curve"] = equity_curve
        return metrics

    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: List[float],
        ticker: str,
        start_date: Any,
        end_date: Any,
    ) -> Dict[str, Any]:
        """İşlemler listesinden kapsamlı portföy ve risk metrikleri üretir."""
        if not trades:
            return {
                "ticker": ticker,
                "total_trades": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "avg_trade_pnl": 0.0,
            }

        df_trades = pd.DataFrame(trades)
        total_trades = len(df_trades)
        wins = df_trades[df_trades["pnl_pct"] > 0]
        losses = df_trades[df_trades["pnl_pct"] < 0]

        win_rate = round((len(wins) / total_trades) * 100, 2)

        gross_profit = wins["pnl_pct"].sum() if not wins.empty else 0.0
        gross_loss = abs(losses["pnl_pct"].sum()) if not losses.empty else 0.0
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        # Toplam Kümülatif Getiri
        initial_eq = equity_curve[0]
        final_eq = equity_curve[-1]
        total_return_pct = round(((final_eq - initial_eq) / initial_eq) * 100, 2)

        # Maksimum Çekilme (Maximum Drawdown - MDD)
        eq_series = pd.Series(equity_curve)
        rolling_max = eq_series.cummax()
        drawdowns = (eq_series - rolling_max) / rolling_max
        max_drawdown_pct = round(abs(drawdowns.min()) * 100, 2)

        # Sharpe Oranı (Yıllıklandırılmış)
        trade_returns = df_trades["pnl_pct"] / 100.0
        mean_ret = trade_returns.mean()
        std_ret = trade_returns.std()
        if std_ret and std_ret > 0:
            # İşlem frekansına göre yaklaşık Sharpe
            sharpe_ratio = round((mean_ret / std_ret) * np.sqrt(52), 2)  # Haftalık ortalama kabulü
        else:
            sharpe_ratio = 0.0

        avg_trade_pnl = round(df_trades["pnl_pct"].mean(), 2)

        # Durum Bazlı İşlem Dağılımı
        status_counts = df_trades["status"].value_counts().to_dict()

        return {
            "ticker": ticker,
            "start_date": str(start_date),
            "end_date": str(end_date),
            "total_trades": total_trades,
            "win_trades": len(wins),
            "loss_trades": len(losses),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "sharpe_ratio": sharpe_ratio,
            "avg_trade_pnl": avg_trade_pnl,
            "status_distribution": status_counts,
        }
