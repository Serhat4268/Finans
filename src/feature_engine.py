"""
Öznitelik Mühendisliği ve İndikatör Hesaplama Modülü (Feature Engine)
Trend, momentum, volatilite, hacim indikatörleri, Supertrend ve sahte kırılım (fakeout) filtreleme metriklerini hesaplar.
"""

import logging
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import ta

from src.config import IndicatorConfig, TargetConfig

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Teknik indikatörlerin ve makine öğrenmesi özniteliklerinin çıkarım motoru."""

    def __init__(
        self,
        ind_config: Optional[IndicatorConfig] = None,
        target_config: Optional[TargetConfig] = None,
    ):
        self.ind_cfg = ind_config or IndicatorConfig()
        self.target_cfg = target_config or TargetConfig()

    def build_features(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """
        Giriş OHLCV DataFrame'ine tüm teknik indikatörleri ve filtreleri ekler.

        Parametreler:
            df (pd.DataFrame): 'Open', 'High', 'Low', 'Close', 'Volume' kolonlarına sahip veri seti.
            include_target (bool): Eğitim için hedef (target) kolonunun hesaplanıp eklenmesi.

        Döndürür:
            pd.DataFrame: Tüm özniteliklerle zenginleştirilmiş veri seti.
        """
        data = df.copy()

        # 1. Trend İndikatörleri (EMA & MACD)
        data = self._add_trend_indicators(data)

        # 2. Momentum İndikatörleri (RSI & Stokastik)
        data = self._add_momentum_indicators(data)

        # 3. Volatilite İndikatörleri (ATR, Bollinger Bantları)
        data = self._add_volatility_indicators(data)

        # 4. Supertrend İndikatörü
        data = self._add_supertrend(data)

        # 5. Hacim İndikatörleri (OBV, Hacim Ortalamaları)
        data = self._add_volume_indicators(data)

        # 6. Sahte Kırılım (Fakeout) ve Mum Dinamiği Filtreleri
        data = self._add_fakeout_and_candle_features(data)

        # 7. Faz 2 Hazırlığı: Sınıflandırma Hedef Değişkeni (Target Labeling)
        if include_target:
            data = self._add_target_labels(data)

        # İndikatör periyotları nedeniyle oluşan ilk NaN değerlerini temizle
        initial_len = len(data)
        data = data.dropna()
        logger.info(f"Öznitelikler oluşturuldu. NaN satırlar temizlendi ({initial_len} -> {len(data)} satır).")

        return data

    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA'lar ve MACD hesaplamaları."""
        close = df["Close"]

        # EMA'lar
        df[f"EMA_{self.ind_cfg.ema_fast}"] = ta.trend.EMAIndicator(close, window=self.ind_cfg.ema_fast).ema_indicator()
        df[f"EMA_{self.ind_cfg.ema_medium}"] = ta.trend.EMAIndicator(close, window=self.ind_cfg.ema_medium).ema_indicator()
        df[f"EMA_{self.ind_cfg.ema_slow}"] = ta.trend.EMAIndicator(close, window=self.ind_cfg.ema_slow).ema_indicator()

        # Fiyatın EMA'lara olan normalize uzaklıkları (Scale-invariant)
        df["Dist_EMA_Fast"] = (close - df[f"EMA_{self.ind_cfg.ema_fast}"]) / df[f"EMA_{self.ind_cfg.ema_fast}"]
        df["Dist_EMA_Medium"] = (close - df[f"EMA_{self.ind_cfg.ema_medium}"]) / df[f"EMA_{self.ind_cfg.ema_medium}"]
        df["Dist_EMA_Slow"] = (close - df[f"EMA_{self.ind_cfg.ema_slow}"]) / df[f"EMA_{self.ind_cfg.ema_slow}"]

        # EMA Trend Kesişimleri (Golden/Death Cross mantığı)
        df["EMA_Trend_Bullish"] = (df[f"EMA_{self.ind_cfg.ema_fast}"] > df[f"EMA_{self.ind_cfg.ema_medium}"]).astype(int)

        # MACD
        macd = ta.trend.MACD(
            close,
            window_fast=self.ind_cfg.macd_fast,
            window_slow=self.ind_cfg.macd_slow,
            window_sign=self.ind_cfg.macd_signal,
        )
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()
        # MACD Histogram ivmesi (momentum artıyor mu azalıyor mu?)
        df["MACD_Hist_Slope"] = df["MACD_Hist"] - df["MACD_Hist"].shift(1)

        return df

    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI ve Stokastik Osilatör hesaplamaları."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # RSI
        rsi_ind = ta.momentum.RSIIndicator(close, window=self.ind_cfg.rsi_period)
        df["RSI"] = rsi_ind.rsi()
        df["RSI_Delta"] = df["RSI"] - df["RSI"].shift(1)
        df["RSI_Overbought"] = (df["RSI"] >= self.ind_cfg.rsi_overbought).astype(int)
        df["RSI_Oversold"] = (df["RSI"] <= self.ind_cfg.rsi_oversold).astype(int)

        # Stokastik Osilatör
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["Stoch_K"] = stoch.stoch()
        df["Stoch_D"] = stoch.stoch_signal()

        return df

    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """ATR ve Bollinger Bantları hesaplamaları."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # ATR (Average True Range)
        atr_ind = ta.volatility.AverageTrueRange(high, low, close, window=self.ind_cfg.atr_period)
        df["ATR"] = atr_ind.average_true_range()
        # Normalized ATR (NATR): Fiyattan bağımsız yüzde cinsinden volatilite
        df["NATR"] = (df["ATR"] / close) * 100

        # Bollinger Bantları
        bb = ta.volatility.BollingerBands(
            close, window=self.ind_cfg.bb_period, window_dev=self.ind_cfg.bb_std_dev
        )
        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
        df["BB_Mid"] = bb.bollinger_mavg()
        df["BB_Width"] = bb.bollinger_wband()
        df["BB_Pband"] = bb.bollinger_pband()  # %B: Fiyatın band içindeki konumu (0: alt bant, 1: üst bant)

        # Volatilite Sıkışması (Bollinger Squeeze) - Patlama öncesi sinyal
        bb_width_ma = df["BB_Width"].rolling(20).mean()
        df["BB_Squeeze"] = (df["BB_Width"] < bb_width_ma * 0.75).astype(int)

        return df

    def _add_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Supertrend İndikatörü:
        Trend takipçisi ve dinamik stop-loss seviyesi üreten ana indikatörlerden biri.
        """
        period = self.ind_cfg.supertrend_period
        multiplier = self.ind_cfg.supertrend_multiplier

        high = df["High"].values
        low = df["Low"].values
        close = df["Close"].values
        n = len(df)

        # True Range ve ATR hesaplama
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr2[0] = tr1[0]
        tr3[0] = tr1[0]
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Basit EMA / RMA ile ATR serisi
        atr = pd.Series(tr).ewm(alpha=1 / period, adjust=False).mean().values

        hl2 = (high + low) / 2
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)

        final_upperband = np.zeros(n)
        final_lowerband = np.zeros(n)
        supertrend = np.zeros(n)
        trend_direction = np.ones(n, dtype=int)  # 1: Bullish / Uptrend, -1: Bearish / Downtrend

        for i in range(1, n):
            # Final Upper Band
            if basic_upperband[i] < final_upperband[i - 1] or close[i - 1] > final_upperband[i - 1]:
                final_upperband[i] = basic_upperband[i]
            else:
                final_upperband[i] = final_upperband[i - 1]

            # Final Lower Band
            if basic_lowerband[i] > final_lowerband[i - 1] or close[i - 1] < final_lowerband[i - 1]:
                final_lowerband[i] = basic_lowerband[i]
            else:
                final_lowerband[i] = final_lowerband[i - 1]

            # Trend Direction
            if trend_direction[i - 1] == 1:
                if close[i] < final_lowerband[i]:
                    trend_direction[i] = -1
                    supertrend[i] = final_upperband[i]
                else:
                    trend_direction[i] = 1
                    supertrend[i] = final_lowerband[i]
            else:
                if close[i] > final_upperband[i]:
                    trend_direction[i] = 1
                    supertrend[i] = final_lowerband[i]
                else:
                    trend_direction[i] = -1
                    supertrend[i] = final_upperband[i]

        df["Supertrend"] = supertrend
        df["Supertrend_Direction"] = trend_direction
        # Fiyatın Supertrend çizgisine normalize mesafesi
        df["Supertrend_Dist"] = (close - supertrend) / close

        return df

    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hacim bazlı teyit indikatörleri."""
        close = df["Close"]
        volume = df["Volume"]

        # OBV (On-Balance Volume)
        obv_ind = ta.volume.OnBalanceVolumeIndicator(close, volume)
        df["OBV"] = obv_ind.on_balance_volume()
        df["OBV_EMA20"] = ta.trend.EMAIndicator(df["OBV"], window=20).ema_indicator()
        df["OBV_Trend"] = (df["OBV"] > df["OBV_EMA20"]).astype(int)

        # Hacim Hareketli Ortalaması ve Hacim Patlama Oranı
        vol_ma = volume.rolling(self.ind_cfg.volume_ma_period).mean()
        df["Volume_MA"] = vol_ma
        df["Volume_Ratio"] = volume / (vol_ma + 1e-8)  # 1.0 üzeri: ortalama üzeri hacim

        return df

    def _add_fakeout_and_candle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sahte Kırılım (Fakeout) Filtreleri ve Mum Fitil Oranları:
        Hacimsiz kırılımları, aşırı alım/satım tuzaklarını ve pinbar redlerini tespit eder.
        """
        high = df["High"]
        low = df["Low"]
        open_p = df["Open"]
        close = df["Close"]
        candle_range = (high - low).replace(0, 1e-8)

        # Mum Gövde ve Gölge/Fitil Oranları
        body = (close - open_p).abs()
        upper_wick = high - np.maximum(open_p, close)
        lower_wick = np.minimum(open_p, close) - low

        df["Candle_Body_Ratio"] = body / candle_range
        df["Upper_Wick_Ratio"] = upper_wick / candle_range
        df["Lower_Wick_Ratio"] = lower_wick / candle_range

        # Sahte Boğa Kırılımı (Bull Trap / Fakeout High):
        # Üst Bollinger Bandını veya önceki 20 günün zirvesini zorlayıp yüksek fitil bırakan ya da hacimsiz kalan barlar
        prev_20_high = high.shift(1).rolling(20).max()
        fake_breakout_high = (
            (high > prev_20_high) & 
            (close < prev_20_high) & 
            (df["Upper_Wick_Ratio"] > 0.45)
        ) | (
            (close > df["BB_High"]) & 
            (df["Volume_Ratio"] < 1.0)  # Hacimsiz kırılım
        )
        df["Fakeout_High_Risk"] = fake_breakout_high.astype(int)

        # Sahte Ayı Kırılımı (Bear Trap / Fakeout Low):
        # Alt Bollinger Bandını veya önceki 20 günün dip seviyesini zorlayıp güçlü alt fitille toparlanan barlar
        prev_20_low = low.shift(1).rolling(20).min()
        fake_breakdown_low = (
            (low < prev_20_low) & 
            (close > prev_20_low) & 
            (df["Lower_Wick_Ratio"] > 0.45)
        ) | (
            (close < df["BB_Low"]) & 
            (df["Volume_Ratio"] < 1.0)
        )
        df["Fakeout_Low_Risk"] = fake_breakdown_low.astype(int)

        return df

    def _add_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Gelecekteki N bar getirisine göre yön etiketleme (Faz 2 için hazırlık):
        1  -> AL (Buy): Gelecek N bardaki getiri > +Eşik
        -1 -> SAT (Sell): Gelecek N bardaki getiri < -Eşik
        0  -> NÖTR (Hold): Eşik aralığında yatay hareket
        """
        window = self.target_cfg.forward_window
        threshold = self.target_cfg.threshold_pct

        # Gelecekteki getiri (Lookahead bias'ı önlemek için sadece etiketleme amaçlıdır)
        future_return = (df["Close"].shift(-window) - df["Close"]) / df["Close"]
        df["Future_Return"] = future_return

        # Sınıf etiketleme
        conditions = [
            future_return > threshold,
            future_return < -threshold,
        ]
        choices = [1, -1]  # 1: Al, -1: Sat
        df["Target"] = np.select(conditions, choices, default=0)

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """Model eğitiminde kullanılacak girdi (X) kolonlarının listesini döndürür."""
        exclude_cols = [
            "Open", "High", "Low", "Close", "Volume",
            "Future_Return", "Target", "Supertrend"
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
