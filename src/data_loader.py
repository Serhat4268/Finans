"""
Veri Çekme ve Temizleme Modülü (Data Loader)
Yahoo Finance (yfinance) üzerinden OHLCV verilerini güvenilir, doğrulanmış ve normalize edilmiş şekilde indirir.
"""

import logging
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class DataLoader:
    """Finansal varlıkların geçmiş ve anlık OHLCV verilerini çeken ve doğrulayan sınıf."""

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            self.cache_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
        else:
            self.cache_dir = Path(cache_dir).resolve()
            
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass  # Dizin zaten varsa veya yetki kısıtlıysa yoksay

    def fetch_data(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """
        yfinance üzerinden varlık verisini çeker.

        Parametreler:
            ticker (str): Varlık kodu (örn: 'BTC-USD', 'AAPL', 'THYAO.IS')
            period (str): Veri aralığı ('1mo', '6mo', '1y', '2y', '5y', 'max')
            interval (str): Zaman dilimi ('1m', '5m', '15m', '1h', '1d')
            start_date (str): 'YYYY-MM-DD' formatında başlangıç tarihi
            end_date (str): 'YYYY-MM-DD' formatında bitiş tarihi
            use_cache (bool): Önceden kaydedilmiş veriyi yerelden oku

        Döndürür:
            pd.DataFrame: Temizlenmiş ve doğrulanmış OHLCV verisi
        """
        cache_file = self.cache_dir / f"{ticker.replace('/', '_')}_{interval}.parquet"

        if use_cache and cache_file.exists():
            logger.info(f"Yerel önbellekten veri yükleniyor: {cache_file}")
            df = pd.read_parquet(cache_file)
            return df

        logger.info(f"Yahoo Finance üzerinden veri indiriliyor: {ticker} (Periyot: {period}, Aralık: {interval})")
        
        try:
            asset = yf.Ticker(ticker)
            if start_date and end_date:
                df = asset.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
            else:
                df = asset.history(period=period, interval=interval, auto_adjust=True)

            if df.empty:
                raise ValueError(f"'{ticker}' için veri çekilemedi. Lütfen sembolü ve tarihleri kontrol edin.")

            # MultiIndex kolon kontrolü (yfinance son sürümlerinde bazen tekil sembolde bile MultiIndex dönebilir)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]

            # Standart kolon adlandırması ve gereksiz kolonların elenmesi
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[col for col in required_cols if col in df.columns]]

            # Veri tipleri ve indeks formatı
            df.index = pd.to_datetime(df.index)
            # Timezone bilgisini UTC'ye eşitleyip timezone-naive yapıyoruz (tarih karşılaştırma tutarlılığı için)
            if df.index.tz is not None:
                df.index = df.index.tz_convert("UTC").tz_localize(None)

            df = self._validate_and_clean(df, ticker)

            # Önbelleğe kaydet
            try:
                df.to_parquet(cache_file)
            except Exception as cache_err:
                logger.warning(f"Önbellek kaydı yapılamadı: {cache_err}")
            logger.info(f"Veri başarıyla indirildi ve temizlendi. Toplam bar sayısı: {len(df)}")
            return df

        except Exception as e:
            logger.error(f"Veri indirme hatası ({ticker}): {str(e)}")
            raise

    def _validate_and_clean(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        OHLCV verisinin finansal tutarlılığını denetler ve temizler:
        - Boş (NaN/Null) değer temizliği
        - Mantıksal tutarsızlık kontrolü (High < Low, High < Open/Close, Low > Open/Close)
        - Negatif veya sıfır fiyat/hacim temizliği
        """
        initial_len = len(df)
        
        # 1. Eksik değerleri temizle / forward fill
        df = df.ffill().bfill()
        
        # 2. Mantıksal fiyat filtreleri
        logical_mask = (
            (df["High"] >= df["Low"]) &
            (df["High"] >= df["Open"]) &
            (df["High"] >= df["Close"]) &
            (df["Low"] <= df["Open"]) &
            (df["Low"] <= df["Close"]) &
            (df["Close"] > 0) &
            (df["Volume"] >= 0)
        )
        
        invalid_rows = (~logical_mask).sum()
        if invalid_rows > 0:
            logger.warning(f"{ticker} verisinde {invalid_rows} adet mantıksız OHLCV satırı tespit edildi ve elendi.")
            df = df[logical_mask]

        # 3. Mükerrer zaman damgalarını kaldır
        df = df[~df.index.duplicated(keep="first")]

        # 4. Zaman serisi artan sırada olmalı
        df = df.sort_index(ascending=True)

        cleaned_len = len(df)
        if cleaned_len < initial_len:
            logger.info(f"{initial_len - cleaned_len} satır temizleme sürecinde ayıklandı.")

        return df


if __name__ == "__main__":
    loader = DataLoader()
    sample_df = loader.fetch_data("BTC-USD", period="1y", interval="1d")
    print(sample_df.tail())
