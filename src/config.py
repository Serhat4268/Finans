"""
Proje Genel Konfigürasyonu
Akademik ve modüler standartlara uygun parametre yönetimi.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class IndicatorConfig:
    # Trend İndikatörleri
    ema_fast: int = 20
    ema_medium: int = 50
    ema_slow: int = 200
    
    # Momentum İndikatörleri
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volatilite İndikatörleri
    atr_period: int = 14
    bb_period: int = 20
    bb_std_dev: float = 2.0
    
    # Supertrend
    supertrend_period: int = 10
    supertrend_multiplier: float = 3.0
    
    # Hacim & Sahte Kırılım (Fakeout) Filtreleri
    volume_ma_period: int = 20
    volume_spike_threshold: float = 1.5  # Hacim hareketli ortalamanın 1.5 katı üzerindeyse geçerli kırılım


@dataclass
class TargetConfig:
    """Etiketleme (Target Labeling) Parametreleri (Faz 2 için hazırlık)"""
    forward_window: int = 3          # N bar sonrası yön tahmini
    threshold_pct: float = 0.015     # %1.5 üzeri yükseliş: Al (1), %1.5 altı düşüş: Sat (-1), aradaki: Nötr (0)


@dataclass
class DataConfig:
    default_ticker: str = "BTC-USD"
    default_period: str = "2y"       # 2 yıllık geçmiş veri
    default_interval: str = "1d"     # Günlük periyot
    supported_assets: List[str] = field(
        default_factory=lambda: ["BTC-USD", "ETH-USD", "AAPL", "NVDA", "THYAO.IS", "EURUSD=X"]
    )
