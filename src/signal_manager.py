"""
SQLite Sinyal Veritabanı ve Karar Loglama Modülü (Signal Manager)
Yapay zeka modellerinin ürettiği yön sinyallerini, olasılık skorlarını ve dinamik
risk yönetimi (ATR stop-loss / take-profit) seviyelerini SQLite veritabanına işler.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class SignalManager:
    """Yapay zeka sinyallerini ve risk parametrelerini SQLite'a kaydeden ve yöneten sınıf."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            self.db_path = Path(__file__).resolve().parent.parent / "database" / "signals.db"
        else:
            self.db_path = Path(db_path).resolve()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Veritabanı bağlantısı döndürür (Row factory ile sözlük benzeri erişim)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Veritabanı tablolarını ve indekslerini oluşturur."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Sinyaller Tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    signal INTEGER NOT NULL,          -- 1: AL, 0: NÖTR, -1: SAT
                    signal_label TEXT NOT NULL,       -- 'AL', 'NOTR', 'SAT'
                    confidence REAL NOT NULL,         -- En yüksek sınıfın olasılığı (0.0 - 1.0)
                    prob_buy REAL NOT NULL,           -- AL olasılığı
                    prob_neutral REAL NOT NULL,       -- NÖTR olasılığı
                    prob_sell REAL NOT NULL,          -- SAT olasılığı
                    entry_price REAL NOT NULL,        -- Sinyal anındaki kapanış fiyatı
                    stop_loss REAL,                   -- Dinamik ATR Stop-Loss fiyatı
                    take_profit REAL,                 -- Dinamik Kâr Al fiyatı (Risk/Reward 1:2)
                    supertrend REAL,                  -- Sinyal anındaki Supertrend seviyesi
                    atr REAL,                         -- Sinyal anındaki ATR değeri
                    status TEXT DEFAULT 'OPEN',       -- 'OPEN', 'HIT_TP', 'HIT_SL', 'EXPIRED', 'HOLD'
                    exit_price REAL,                  -- Pozisyondan çıkış fiyatı
                    exit_timestamp TEXT,              -- Çıkış zaman damgası
                    pnl_pct REAL,                     -- Yüzde kâr/zarar (% örn: +3.5 veya -1.8)
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 2. Backtest Performans Kayıtları Tablosu
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    total_trades INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    profit_factor REAL,
                    total_return_pct REAL NOT NULL,
                    max_drawdown_pct REAL NOT NULL,
                    sharpe_ratio REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # İndeksler
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_ticker ON signals(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status)")
            conn.commit()

    @staticmethod
    def calculate_trade_levels(
        signal_raw: int,
        close: float,
        atr: float,
        atr_multiplier: float = 1.5,
        risk_reward_ratio: float = 2.0,
        spot_only: bool = True,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Dinamik ATR tabanlı Stop-Loss ve Take-Profit (Hedef) seviyelerini hesaplar.
        Spot işlem modelinde sadece AL (1) için pozisyon seviyeleri belirlenir;
        kaldıraçsız spot piyasada SAT veya NÖTR iken yeni pozisyon açılmaz (nakitte beklenir).
        """
        if signal_raw == 1:  # AL (Spot Alış)
            stop_loss = round(close - (atr * atr_multiplier), 4)
            take_profit = round(close + (atr * atr_multiplier * risk_reward_ratio), 4)
            return stop_loss, take_profit

        elif signal_raw == -1 and not spot_only:  # Kaldıraçlı/Short Açığa Satış
            stop_loss = round(close + (atr * atr_multiplier), 4)
            take_profit = round(close - (atr * atr_multiplier * risk_reward_ratio), 4)
            return stop_loss, take_profit

        return None, None  # Spot piyasada SAT veya NÖTR durumunda pozisyon açılmaz

    def log_signal(
        self,
        ticker: str,
        timestamp: str,
        signal_raw: int,
        signal_label: str,
        confidence: float,
        probabilities: Dict[str, float],
        entry_price: float,
        stop_loss: Optional[float],
        take_profit: Optional[float],
        supertrend: float,
        atr: float,
        status: str = "OPEN",
    ) -> int:
        """Üretilen yeni bir sinyali SQLite'a kaydeder ve sinyal ID'sini döner. Mükerrer kayıtları önler."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Mükerrer kayıt kontrolü (aynı sembol ve bar zamanı için zaten varsa ID döner)
            cursor.execute(
                "SELECT id FROM signals WHERE ticker = ? AND timestamp = ?",
                (ticker, str(timestamp)),
            )
            existing = cursor.fetchone()
            if existing:
                return int(existing[0])

            cursor.execute(
                """
                INSERT INTO signals (
                    ticker, timestamp, signal, signal_label, confidence,
                    prob_buy, prob_neutral, prob_sell,
                    entry_price, stop_loss, take_profit,
                    supertrend, atr, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    str(timestamp),
                    signal_raw,
                    signal_label,
                    confidence,
                    probabilities.get("AL", 0.0),
                    probabilities.get("NOTR", 0.0),
                    probabilities.get("SAT", 0.0),
                    entry_price,
                    stop_loss,
                    take_profit,
                    supertrend,
                    atr,
                    status if (signal_raw == 1 and status == "OPEN") else "HOLD",
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def update_signal_exit(
        self,
        signal_id: int,
        status: str,
        exit_price: float,
        exit_timestamp: str,
        pnl_pct: float,
    ):
        """Sonuçlanan bir pozisyonun kâr/zarar ve çıkış durumunu günceller."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE signals
                SET status = ?, exit_price = ?, exit_timestamp = ?, pnl_pct = ?
                WHERE id = ?
                """,
                (status, exit_price, str(exit_timestamp), round(pnl_pct, 4), signal_id),
            )
            conn.commit()

    def get_signals(
        self, ticker: Optional[str] = None, limit: int = 100, status: Optional[str] = None
    ) -> pd.DataFrame:
        """Veritabanındaki sinyalleri DataFrame olarak çeker."""
        query = "SELECT * FROM signals WHERE 1=1"
        params: List[Any] = []

        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            return df

    def get_latest_signal(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Belirtilen varlık için kaydedilen en son sinyali sözlük olarak döner."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM signals WHERE ticker = ? ORDER BY id DESC LIMIT 1",
                (ticker,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def log_backtest_run(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        total_trades: int,
        win_rate: float,
        profit_factor: Optional[float],
        total_return_pct: float,
        max_drawdown_pct: float,
        sharpe_ratio: Optional[float],
    ) -> int:
        """Backtest özet metriklerini kaydeder."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO backtest_runs (
                    ticker, start_date, end_date, total_trades, win_rate,
                    profit_factor, total_return_pct, max_drawdown_pct, sharpe_ratio
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    start_date,
                    end_date,
                    total_trades,
                    win_rate,
                    profit_factor,
                    total_return_pct,
                    max_drawdown_pct,
                    sharpe_ratio,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_live_signals(
        self,
        ticker: str,
        start_date: str,
    ) -> pd.DataFrame:
        """
        Belirtilen başlangıç tarihinden (dahil) itibaren kaydedilen tüm sinyalleri döndürür.
        İleriye dönük (forward-looking) sermaye takibi için kullanılır.

        Parametreler:
            ticker: Varlık sembolü.
            start_date: 'YYYY-MM-DD' formatında başlangıç tarihi (bu tarih dahil).

        Döndürür:
            pd.DataFrame: Başlangıç tarihinden sonraki tüm sinyal kayıtları.
        """
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                """
                SELECT *
                FROM signals
                WHERE ticker = ?
                  AND DATE(timestamp) >= DATE(?)
                ORDER BY id ASC
                """,
                conn,
                params=[ticker, start_date],
            )
            return df

    def resolve_open_positions(
        self,
        ticker: str,
        df_bars: pd.DataFrame,
        max_holding_bars: int = 10,
    ) -> int:
        """
        Durum 'OPEN' olan sinyaller için güncel fiyat barlarına bakarak
        Stop-Loss veya Take-Profit tetiklenip tetiklenmediğini kontrol eder;
        tetiklendiyse veya maksimum bekleme barı (max_holding_bars) dolduysa pozisyonu kapatır.

        Mantık:
          - AL (signal=1): bar'ın Low <= stop_loss => HIT_SL, High >= take_profit => HIT_TP
          - SAT (signal=-1): bar'ın High >= stop_loss => HIT_SL, Low <= take_profit => HIT_TP
          - Ne SL ne TP vurulmadıysa ve len(future_bars) >= max_holding_bars ise EXPIRED olarak kapat.
          - Süre dolmadıysa pozisyon 'OPEN' kalır ve anlık izlenir.

        Parametreler:
            ticker: Varlık sembolü.
            df_bars: FeatureEngine'in ürettiği en güncel OHLCV DataFrame'i (tarih indeksli).
            max_holding_bars: Pozisyonun açık kalabileceği azami bar sayısı (varsayılan: 10).

        Döndürür:
            Güncellenen (kapatılan) pozisyon sayısı.
        """
        open_signals = self.get_signals(ticker=ticker, status="OPEN")
        if open_signals.empty:
            return 0

        # Spot modelinde sadece AL (signal=1) pozisyonları izlenir
        open_signals = open_signals[open_signals["signal"] == 1]
        if open_signals.empty:
            return 0

        updated_count = 0
        df_bars_sorted = df_bars.sort_index()

        for _, sig_row in open_signals.iterrows():
            sig_id = int(sig_row["id"])
            sig_direction = int(sig_row["signal"])
            entry_ts = pd.to_datetime(str(sig_row["timestamp"])).tz_localize(None)
            stop_loss = sig_row["stop_loss"]
            take_profit = sig_row["take_profit"]
            entry_price = float(sig_row["entry_price"])

            if stop_loss is None or take_profit is None:
                continue

            stop_loss = float(stop_loss)
            take_profit = float(take_profit)

            # Giriş tarihinden SONRA gelen barları al (ilk bar giriş barı, ona bakma)
            future_bars = df_bars_sorted[df_bars_sorted.index > entry_ts]
            if future_bars.empty:
                continue  # Henüz yeni bar gelmemiş, pozisyon açık kalmaya devam eder

            exit_found = False
            # Yalnızca max_holding_bars kadar barı tara
            eval_bars = future_bars.iloc[:max_holding_bars]
            for bar_time, bar in eval_bars.iterrows():
                high_p = float(bar["High"])
                low_p = float(bar["Low"])
                close_p = float(bar["Close"])

                if sig_direction == 1:  # AL (Long)
                    if low_p <= stop_loss:
                        status = "HIT_SL"
                        exit_price = stop_loss
                        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                        self.update_signal_exit(sig_id, status, exit_price, str(bar_time), round(pnl_pct, 4))
                        exit_found = True
                        updated_count += 1
                        break
                    elif high_p >= take_profit:
                        status = "HIT_TP"
                        exit_price = take_profit
                        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                        self.update_signal_exit(sig_id, status, exit_price, str(bar_time), round(pnl_pct, 4))
                        exit_found = True
                        updated_count += 1
                        break

                elif sig_direction == -1:  # SAT (Short)
                    if high_p >= stop_loss:
                        status = "HIT_SL"
                        exit_price = stop_loss
                        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                        self.update_signal_exit(sig_id, status, exit_price, str(bar_time), round(pnl_pct, 4))
                        exit_found = True
                        updated_count += 1
                        break
                    elif low_p <= take_profit:
                        status = "HIT_TP"
                        exit_price = take_profit
                        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                        self.update_signal_exit(sig_id, status, exit_price, str(bar_time), round(pnl_pct, 4))
                        exit_found = True
                        updated_count += 1
                        break

            if not exit_found:
                # Eğer maksimum bekleme süresi (max_holding_bars) dolduysa EXPIRED olarak kapat
                if len(future_bars) >= max_holding_bars:
                    last_bar = eval_bars.iloc[-1]
                    exit_price = float(last_bar["Close"])
                    last_time = str(eval_bars.index[-1])
                    if sig_direction == 1:
                        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                    else:
                        pnl_pct = ((entry_price - exit_price) / entry_price) * 100
                    self.update_signal_exit(sig_id, "EXPIRED", exit_price, last_time, round(pnl_pct, 4))
                    updated_count += 1
                else:
                    # Bekleme süresi dolmadı ve SL/TP vurulmadı: Pozisyon açık kalmaya devam eder!
                    pass

        return updated_count

    def cleanup_duplicate_signals(self, ticker: Optional[str] = None) -> int:
        """Aynı sembol ve zaman damgasına sahip mükerrer kayıtları temizler, en küçük ID'yi korur."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if ticker:
                cursor.execute("""
                    DELETE FROM signals
                    WHERE id NOT IN (
                        SELECT MIN(id)
                        FROM signals
                        WHERE ticker = ?
                        GROUP BY ticker, timestamp
                    ) AND ticker = ?
                """, (ticker, ticker))
            else:
                cursor.execute("""
                    DELETE FROM signals
                    WHERE id NOT IN (
                        SELECT MIN(id)
                        FROM signals
                        GROUP BY ticker, timestamp
                    )
                """)
            conn.commit()
            return cursor.rowcount

    def clear_database(self, ticker: Optional[str] = None):
        """Veritabanındaki sinyal ve test kayıtlarını temizler (Testler için)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if ticker:
                cursor.execute("DELETE FROM signals WHERE ticker = ?", (ticker,))
                cursor.execute("DELETE FROM backtest_runs WHERE ticker = ?", (ticker,))
            else:
                cursor.execute("DELETE FROM signals")
                cursor.execute("DELETE FROM backtest_runs")
            conn.commit()
