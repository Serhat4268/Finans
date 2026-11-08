"""
Makine Öğrenmesi ile Teknik İndikatör Tabanlı Finansal Varlık Yön Tahmini ve Gerçek Zamanlı Sinyal Üretim Platformu
TradingView / Bloomberg Terminal Seviyesinde Ultra Modern Karar Destek Arayüzü
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Proje ana dizinini path'e ekle
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from src.data_loader import DataLoader
from src.feature_engine import FeatureEngine
from src.model_trainer import FinancialModelTrainer
from src.signal_manager import SignalManager
from src.backtester import BacktestEngine
from src.config import DataConfig

# ==============================================================================
# Sayfa Genel Ayarları
# ==============================================================================
st.set_page_config(
    page_title="Makine Öğrenmesi Karar Destek Platformu",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# Özel CSS: Bloomberg / TradingView Dark Terminal Tasarımı
# ==============================================================================
st.markdown("""
<style>
    /* Streamlit Deploy Butonunu ve Sağ Menüyü Kaldır */
    .stDeployButton,
    [data-testid="stDeployButton"],
    [data-testid="stToolbarActions"],
    [data-testid="stMainMenu"] {
        display: none !important;
    }
    
    /* Sol Kenar Çubuğunu Açma Butonunu (Expand Sidebar) Her Zaman Görünür ve Şık Yap */
    [data-testid="stExpandSidebarButton"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        color: #58a6ff !important;
        padding: 4px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stExpandSidebarButton"]:hover {
        background: #21262d !important;
        border-color: #58a6ff !important;
        box-shadow: 0 0 8px rgba(88, 166, 255, 0.4) !important;
    }
    
    /* Ana Arka Plan ve Tipografi */
    .stApp {
        background-color: #0d1117;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Üst Başlık Banner */
    .header-banner {
        background: linear-gradient(135deg, #161b22 0%, #1a2332 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    /* Modern Metrik Kartları */
    .kpi-card {
        background: linear-gradient(180deg, #161b22 0%, #12161c 100%);
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, border-color 0.2s ease;
        height: 100%;
    }
    .kpi-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .kpi-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b949e;
        margin-bottom: 6px;
        font-weight: 600;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #f0f6fc;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.82rem;
        margin-top: 6px;
        font-weight: 500;
    }
    
    /* Canlı Parlayan Karar Rozetleri (Glow Effect) */
    .badge-buy {
        display: inline-block;
        background: linear-gradient(135deg, #00c853 0%, #009624 100%);
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: 0.05em;
        box-shadow: 0 0 16px rgba(0, 200, 83, 0.5);
    }
    .badge-sell {
        display: inline-block;
        background: linear-gradient(135deg, #ff1744 0%, #c51162 100%);
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: 0.05em;
        box-shadow: 0 0 16px rgba(255, 23, 68, 0.5);
    }
    .badge-hold {
        display: inline-block;
        background: linear-gradient(135deg, #ff9100 0%, #ff6d00 100%);
        color: #ffffff;
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.25rem;
        letter-spacing: 0.05em;
        box-shadow: 0 0 16px rgba(255, 145, 0, 0.4);
    }
    
    /* Tek Parça Yığılmış Olasılık Çubuğu */
    .prob-bar-wrapper {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 15px 0 25px 0;
    }
    .prob-bar-container {
        display: flex;
        width: 100%;
        height: 24px;
        border-radius: 6px;
        overflow: hidden;
        margin: 8px 0;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.5);
    }
    .prob-segment-buy {
        background: linear-gradient(90deg, #26a69a, #00c853);
        transition: width 0.4s ease;
    }
    .prob-segment-hold {
        background: linear-gradient(90deg, #ffa726, #ff9100);
        transition: width 0.4s ease;
    }
    .prob-segment-sell {
        background: linear-gradient(90deg, #ef5350, #d50000);
        transition: width 0.4s ease;
    }
    
    /* Durum Hapları (Pills) */
    .pill-green { color: #26a69a; font-weight: bold; }
    .pill-red { color: #ef5350; font-weight: bold; }
    .pill-orange { color: #ffa726; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Önbellek Fonksiyonları
# ==============================================================================
@st.cache_data(ttl=300)
def fetch_asset_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Veriyi Yahoo Finance üzerinden çeker veya Parquet önbellekten okur."""
    loader = DataLoader()
    return loader.fetch_data(ticker=ticker, period=period, interval=interval, use_cache=True)


def get_model_bundle(ticker: str, model_type: str, raw_df: pd.DataFrame, feature_engine: FeatureEngine) -> dict:
    """Kayıtlı modeli yükler; yoksa hızlıca eğitip .pkl paketini oluşturur."""
    safe_ticker = ticker.replace("-", "_").replace("=", "_").replace("/", "_")
    model_path = ROOT_DIR / "models" / f"model_{safe_ticker}_{model_type}.pkl"

    trainer = FinancialModelTrainer()
    if model_path.exists():
        return FinancialModelTrainer.load_model_bundle(model_path)
    
    with st.spinner(f"'{ticker}' için {model_type.upper()} modeli eğitiliyor (Time-Series Split)..."):
        df_train = feature_engine.build_features(raw_df, include_target=True)
        features = feature_engine.get_feature_columns(df_train)
        X, y = trainer.prepare_data(df_train, features)
        cv_res = trainer.cross_validate(X, y, model_type=model_type)
        final_model = trainer.train_final_model(X, y, model_type=model_type)
        trainer.save_model_bundle(
            model=final_model,
            feature_names=features,
            ticker=ticker,
            model_type=model_type,
            metrics=cv_res,
        )
        return FinancialModelTrainer.load_model_bundle(model_path)


# ==============================================================================
# Plotly TradingView Seviyesinde İnteraktif Şamdan ve Sinyal Grafiği
# ==============================================================================
def create_terminal_chart(
    df: pd.DataFrame,
    ticker: str,
    show_supertrend: bool = True,
    show_bollinger: bool = True,
    show_ema: bool = True,
    show_signals: bool = True,
    bundle: dict = None,
) -> go.Figure:
    """TradingView kalitesinde 3 panelli teknik grafik ve sinyal okları."""
    plot_df = df.tail(130).copy()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.62, 0.18, 0.20],
        subplot_titles=(
            f"<b>{ticker}</b> Fiyat Dinamikleri & Teknik İndikatörler",
            "İşlem Hacmi (Volume) & 20 MA",
            "RSI (14) Momentum Göstergesi",
        ),
    )

    # 1. Şamdan Grafiği (Candlestick)
    fig.add_trace(
        go.Candlestick(
            x=plot_df.index,
            open=plot_df["Open"],
            high=plot_df["High"],
            low=plot_df["Low"],
            close=plot_df["Close"],
            name="OHLC",
            increasing=dict(line=dict(color="#26a69a", width=1), fillcolor="#26a69a"),
            decreasing=dict(line=dict(color="#ef5350", width=1), fillcolor="#ef5350"),
        ),
        row=1, col=1,
    )

    # 2. Bollinger Bantları
    if show_bollinger and "BB_High" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["BB_High"],
                line=dict(color="rgba(88, 166, 255, 0.45)", width=1, dash="dot"),
                name="BB Üst (Direnç)",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["BB_Low"],
                line=dict(color="rgba(88, 166, 255, 0.45)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(88, 166, 255, 0.04)",
                name="BB Alt (Destek)",
            ),
            row=1, col=1,
        )

    # 3. EMA 20 & EMA 50
    if show_ema:
        if "EMA_20" in plot_df.columns:
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["EMA_20"], line=dict(color="#00e5ff", width=1.5), name="EMA 20 (Hızlı Trend)"),
                row=1, col=1,
            )
        if "EMA_50" in plot_df.columns:
            fig.add_trace(
                go.Scatter(x=plot_df.index, y=plot_df["EMA_50"], line=dict(color="#ffa726", width=1.5), name="EMA 50 (Orta Trend)"),
                row=1, col=1,
            )

    # 4. Supertrend Çizgisi
    if show_supertrend and "Supertrend" in plot_df.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_df.index, y=plot_df["Supertrend"],
                line=dict(color="#d500f9", width=2),
                name="Supertrend Bandı",
            ),
            row=1, col=1,
        )

    # 5. Model Sinyal Okları (Markers on Chart)
    if show_signals and bundle is not None:
        model = bundle["model"]
        trained_feats = bundle["feature_names"]
        
        preds_enc = model.predict(plot_df[trained_feats])
        probs = model.predict_proba(plot_df[trained_feats])
        
        # AL (Buy) Okları (Yeşil Yukarı Ok)
        buy_mask = (preds_enc == 2)
        if buy_mask.any():
            buy_times = plot_df.index[buy_mask]
            buy_prices = plot_df.loc[buy_mask, "Low"] * 0.985
            buy_confs = np.max(probs[buy_mask], axis=1) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=buy_times, y=buy_prices,
                    mode="markers+text",
                    marker=dict(symbol="triangle-up", size=13, color="#00e676", line=dict(width=1, color="#ffffff")),
                    text=["AL" for _ in buy_times],
                    textposition="bottom center",
                    textfont=dict(size=10, color="#00e676"),
                    hovertext=[f"AL Sinyali<br>Güven: %{c:.1f}<br>Tarih: {str(t)[:10]}" for c, t in zip(buy_confs, buy_times)],
                    hoverinfo="text",
                    name="ML AL Sinyali",
                ),
                row=1, col=1,
            )

        # SAT (Sell) Okları (Kırmızı Aşağı Ok)
        sell_mask = (preds_enc == 0)
        if sell_mask.any():
            sell_times = plot_df.index[sell_mask]
            sell_prices = plot_df.loc[sell_mask, "High"] * 1.015
            sell_confs = np.max(probs[sell_mask], axis=1) * 100
            
            fig.add_trace(
                go.Scatter(
                    x=sell_times, y=sell_prices,
                    mode="markers+text",
                    marker=dict(symbol="triangle-down", size=13, color="#ff1744", line=dict(width=1, color="#ffffff")),
                    text=["SAT" for _ in sell_times],
                    textposition="top center",
                    textfont=dict(size=10, color="#ff1744"),
                    hovertext=[f"SAT Sinyali<br>Güven: %{c:.1f}<br>Tarih: {str(t)[:10]}" for c, t in zip(sell_confs, sell_times)],
                    hoverinfo="text",
                    name="ML SAT Sinyali",
                ),
                row=1, col=1,
            )

    # 6. Hacim (Volume) ve Hacim MA
    vol_colors = ["#26a69a" if c >= o else "#ef5350" for c, o in zip(plot_df["Close"], plot_df["Open"])]
    fig.add_trace(
        go.Bar(x=plot_df.index, y=plot_df["Volume"], marker_color=vol_colors, name="Hacim", opacity=0.85),
        row=2, col=1,
    )
    if "Volume_MA" in plot_df.columns:
        fig.add_trace(
            go.Scatter(x=plot_df.index, y=plot_df["Volume_MA"], line=dict(color="#ffd600", width=1.5), name="Hacim 20 MA"),
            row=2, col=1,
        )

    # 7. RSI (14) & Koridor
    fig.add_trace(
        go.Scatter(x=plot_df.index, y=plot_df["RSI"], line=dict(color="#ab47bc", width=1.8), name="RSI (14)"),
        row=3, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 23, 68, 0.65)", annotation_text="Aşırı Alım (70)", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="rgba(0, 230, 118, 0.65)", annotation_text="Aşırı Satım (30)", row=3, col=1)

    # Stil, Izgara ve Crosshair Ayarları
    fig.update_layout(
        template="plotly_dark",
        height=780,
        margin=dict(l=35, r=35, t=50, b=30),
        plot_bgcolor="#0d1117",
        paper_bgcolor="#0d1117",
        xaxis_rangeslider_visible=False,
        dragmode="pan",           # Sol tık sürükle = kaydır (dikdörtgen seçim yok)
        selectdirection="h",      # Seçim yönünü yatay kısıtla (yedek güvenlik)
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11),
            bgcolor="rgba(22, 27, 34, 0.7)",
            bordercolor="#30363d",
            borderwidth=1,
        ),
        hovermode="x unified",
    )

    # Hassas ızgara çizgileri
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)", spikemode="across", spikesnap="cursor", spikedash="dot", spikethickness=1)
    # fixedrange=False → Y ekseni drag handle ile bağımsız ölçeklenebilir (TradingView fiyat skalası davranışı)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(255, 255, 255, 0.05)", spikemode="across", spikesnap="cursor", spikedash="dot", spikethickness=1, fixedrange=False)

    return fig


# ==============================================================================
# Ana Arayüz (Dashboard)
# ==============================================================================
def main():
    # --------------------------------------------------------------------------
    # 1. Kenar Çubuğu (Sidebar)
    # --------------------------------------------------------------------------
    st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 15px;">
            <span style="font-size: 2.8rem;">⚡</span>
            <h3 style="color: #f0f6fc; margin: 4px 0 0 0; font-weight: 700;">ML Karar Paneli</h3>
            <span style="font-size: 0.75rem; color: #58a6ff; letter-spacing: 0.05em; font-weight: 600;">CANLI FİNANSAL VERİ AKIŞI</span>
        </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("---")

    data_cfg = DataConfig()
    selected_ticker = st.sidebar.selectbox("🎯 Finansal Varlık", data_cfg.supported_assets, index=0)
    selected_period = st.sidebar.selectbox("📅 Geçmiş Veri Derinliği", ["6mo", "1y", "2y", "5y"], index=2)
    selected_model = st.sidebar.selectbox("🤖 ML Algoritması", ["lightgbm", "xgboost"], index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #f0f6fc; font-size: 0.95rem; margin-bottom: 8px;'>⚙️ Dinamik Risk Parametreleri</h4>", unsafe_allow_html=True)
    atr_multiplier = st.sidebar.slider("Dinamik Stop Çarpanı (ATR)", 1.0, 3.0, 1.5, 0.1, help="Stop-loss mesafesini volatiliteye göre belirler.")
    risk_reward = st.sidebar.slider("Risk/Ödül Oranı (Take-Profit)", 1.0, 3.5, 2.0, 0.5, help="1 birim riske karşılık hedeflenen kâr katsayısı.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("<h4 style='color: #f0f6fc; font-size: 0.95rem; margin-bottom: 8px;'>👁️ Grafik Katmanları</h4>", unsafe_allow_html=True)
    show_supertrend = st.sidebar.checkbox("Supertrend Bandı", value=True)
    show_bollinger = st.sidebar.checkbox("Bollinger Bantları", value=True)
    show_ema = st.sidebar.checkbox("EMA 20 & 50", value=True)
    show_signals = st.sidebar.checkbox("Model Sinyal Okları (Al/Sat)", value=True)

    st.sidebar.markdown("---")
    refresh_btn = st.sidebar.button("🔄 Verileri & Sinyalleri Yenile", use_container_width=True)

    # --------------------------------------------------------------------------
    # 2. Veri Çekme & Model Yükleme
    # --------------------------------------------------------------------------
    try:
        raw_df = fetch_asset_data(selected_ticker, selected_period, "1d")
        feature_engine = FeatureEngine()
        df_features = feature_engine.build_features(raw_df, include_target=False)
        bundle = get_model_bundle(selected_ticker, selected_model, raw_df, feature_engine)
    except Exception as e:
        st.error(f"Veri çekme veya model çalıştırma hatası: {e}")
        return

    # Sinyal Üretimi
    latest_signal = FinancialModelTrainer.predict_signal(bundle, df_features)
    latest_bar = df_features.iloc[-1]
    prev_bar = df_features.iloc[-2]

    last_close = float(latest_bar["Close"])
    day_high = float(latest_bar["High"])
    day_low = float(latest_bar["Low"])
    price_change = last_close - float(prev_bar["Close"])
    price_change_pct = (price_change / float(prev_bar["Close"])) * 100
    last_atr = float(latest_bar["ATR"])
    last_supertrend = float(latest_bar["Supertrend"])
    supertrend_dir = int(latest_bar["Supertrend_Direction"])

    stop_loss, take_profit = SignalManager.calculate_trade_levels(
        signal_raw=latest_signal["signal_raw"],
        close=last_close,
        atr=last_atr,
        atr_multiplier=atr_multiplier,
        risk_reward_ratio=risk_reward,
    )

    # SQLite Sinyal Kaydı
    signal_mgr = SignalManager()
    signal_mgr.log_signal(
        ticker=selected_ticker,
        timestamp=str(df_features.index[-1]),
        signal_raw=latest_signal["signal_raw"],
        signal_label=latest_signal["signal_label"],
        confidence=latest_signal["confidence"],
        probabilities=latest_signal["probabilities"],
        entry_price=last_close,
        stop_loss=stop_loss,
        take_profit=take_profit,
        supertrend=last_supertrend,
        atr=last_atr,
    )

    # --------------------------------------------------------------------------
    # 3. Üst Başlık Banner
    # --------------------------------------------------------------------------
    st.markdown(f"""
        <div class="header-banner">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <h1 style="color: #f0f6fc; font-size: 1.85rem; margin: 0; font-weight: 800;">
                        📈 {selected_ticker} &mdash; Makine Öğrenmesi Tabanlı Yön Tahmin Platformu
                    </h1>
                    <p style="color: #8b949e; font-size: 0.9rem; margin: 6px 0 0 0;">
                        Algoritmik Teknik Analiz, Volatilite Filtreleme ve Dinamik Risk/Sinyal Karar Destek Sistemi
                    </p>
                </div>
                <div style="text-align: right; margin-top: 6px;">
                    <span style="background: rgba(88, 166, 255, 0.15); color: #58a6ff; padding: 6px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 700; border: 1px solid rgba(88, 166, 255, 0.3);">
                        ● CANLI MODEL AKTİF
                    </span>
                    <div style="color: #8b949e; font-size: 0.75rem; margin-top: 4px;">Zaman Damgası: {str(df_features.index[-1])[:10]}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 4. Üst KPI Metrik Kartları
    # --------------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    delta_color = "#26a69a" if price_change_pct >= 0 else "#ef5350"
    delta_icon = "▲" if price_change_pct >= 0 else "▼"

    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">{selected_ticker} Güncel Fiyat</div>
                <div class="kpi-value">${last_close:,.2f}</div>
                <div class="kpi-sub" style="color: {delta_color};">
                    {delta_icon} {price_change_pct:+.2f}% (${price_change:+,.2f})
                </div>
                <div style="font-size: 0.72rem; color: #8b949e; margin-top: 4px;">
                    Günün Aralığı: ${day_low:,.1f} - ${day_high:,.1f}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        sig_lbl = latest_signal["signal_label"]
        if sig_lbl == "AL":
            badge_html = "<span class='badge-buy'>🚀 AL (BUY)</span>"
            sub_text = "Yükseliş Beklentisi"
            sub_color = "#00c853"
        elif sig_lbl == "SAT":
            badge_html = "<span class='badge-sell'>🔻 SAT (SELL)</span>"
            sub_text = "Düşüş Beklentisi"
            sub_color = "#ff1744"
        else:
            badge_html = "<span class='badge-hold'>⏸️ NÖTR (HOLD)</span>"
            sub_text = "Yatay / Kararsız Piyasa"
            sub_color = "#ff9100"

        st.markdown(f"""
            <div class="kpi-card" style="text-align: center;">
                <div class="kpi-title">Makine Öğrenmesi Tahmini</div>
                <div style="margin-top: 4px;">{badge_html}</div>
                <div class="kpi-sub" style="color: {sub_color}; margin-top: 8px;">{sub_text}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        conf_pct = latest_signal["confidence"] * 100
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Model Güven Skoru</div>
                <div class="kpi-value" style="color: #58a6ff;">%{conf_pct:.1f}</div>
                <div class="kpi-sub" style="color: #8b949e;">Algoritma: {selected_model.upper()}</div>
                <div style="font-size: 0.72rem; color: #58a6ff; margin-top: 4px;">En yüksek sınıf olasılığı</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        if stop_loss:
            sl_pct = abs((stop_loss - last_close) / last_close) * 100
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Dinamik Stop-Loss</div>
                    <div class="kpi-value" style="color: #ef5350;">${stop_loss:,.2f}</div>
                    <div class="kpi-sub" style="color: #ef5350;">-%{sl_pct:.2f} Risk Mesafesi</div>
                    <div style="font-size: 0.72rem; color: #8b949e; margin-top: 4px;">{atr_multiplier:.1f}x ATR Bazlı Dinamik Seviye</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="kpi-card">
                    <div class="kpi-title">Dinamik Stop-Loss</div>
                    <div class="kpi-value" style="color: #8b949e;">Beklemede</div>
                    <div class="kpi-sub" style="color: #8b949e;">Pozisyon Açılmadı (Nötr)</div>
                </div>
            """, unsafe_allow_html=True)

    with col5:
        if take_profit:
            tp_pct = abs((take_profit - last_close) / last_close) * 100
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-title">Kâr Al Hedefi (Take-Profit)</div>
                    <div class="kpi-value" style="color: #26a69a;">${take_profit:,.2f}</div>
                    <div class="kpi-sub" style="color: #26a69a;">+%{tp_pct:.2f} Kâr Potansiyeli</div>
                    <div style="font-size: 0.72rem; color: #8b949e; margin-top: 4px;">Hedef Risk/Ödül: 1:{risk_reward:.1f}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="kpi-card">
                    <div class="kpi-title">Kâr Al Hedefi</div>
                    <div class="kpi-value" style="color: #8b949e;">Beklemede</div>
                    <div class="kpi-sub" style="color: #8b949e;">Pozisyon Açılmadı (Nötr)</div>
                </div>
            """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 5. Sahte Kırılım (Fakeout) Banner'ı
    # --------------------------------------------------------------------------
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    if int(latest_bar.get("Fakeout_High_Risk", 0)) == 1:
        st.warning("⚠️ **DİKKAT (BOĞA TUZAĞI RİSKİ):** Fiyat direnç kırdı ancak mum uzun bir üst fitil bıraktı veya hacim ortalamanın altında! Sahte kırılım riski yüksek.")
    elif int(latest_bar.get("Fakeout_Low_Risk", 0)) == 1:
        st.info("ℹ️ **BİLGİ (AYI TUZAĞI / DİP TOPARLANMASI):** Fiyat destek bandını zorladı ancak güçlü alt fitille toparlandı. Sahte çöküş riski mevcut.")

    # --------------------------------------------------------------------------
    # 6. Tek Parça Yığılmış Olasılık Çubuğu (TradingView Tarzı)
    # --------------------------------------------------------------------------
    p_al = latest_signal["probabilities"].get("AL", 0.0) * 100
    p_notr = latest_signal["probabilities"].get("NOTR", 0.0) * 100
    p_sat = latest_signal["probabilities"].get("SAT", 0.0) * 100

    st.markdown(f"""
        <div class="prob-bar-wrapper">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span style="font-size: 0.88rem; font-weight: 700; color: #f0f6fc;">
                    🎯 Model Yön Olasılık Dağılımı (Kuvvet Dengesi)
                </span>
                <span style="font-size: 0.8rem; color: #8b949e;">Toplam: %100</span>
            </div>
            <div class="prob-bar-container">
                <div class="prob-segment-sell" style="width: {p_sat}%;" title="Düşüş: %{p_sat:.1f}"></div>
                <div class="prob-segment-hold" style="width: {p_notr}%;" title="Yatay: %{p_notr:.1f}"></div>
                <div class="prob-segment-buy" style="width: {p_al}%;" title="Yükseliş: %{p_al:.1f}"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-top: 6px;">
                <span style="color: #ef5350; font-weight: 700;">🔴 Düşüş: %{p_sat:.1f}</span>
                <span style="color: #ffa726; font-weight: 700;">🟡 Yatay / Kararsız: %{p_notr:.1f}</span>
                <span style="color: #26a69a; font-weight: 700;">🟢 Yükseliş: %{p_al:.1f}</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 7. İnteraktif Plotly Şamdan ve Sinyal Grafiği
    # --------------------------------------------------------------------------
    fig = create_terminal_chart(
        df=df_features,
        ticker=selected_ticker,
        show_supertrend=show_supertrend,
        show_bollinger=show_bollinger,
        show_ema=show_ema,
        show_signals=show_signals,
        bundle=bundle,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,               # Scroll tekerleği = zoom
            "displayModeBar": True,           # Araç çubuğu görünür kalsın
            "modeBarButtonsToRemove": [       # Dikdörtgen seçim ve lasso kaldırıldı
                "select2d",
                "lasso2d",
                "autoScale2d",
            ],
        },
    )

    # --------------------------------------------------------------------------
    # 8. Sekmeli Detay Paneli (Tabs)
    # --------------------------------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 SQLite Sinyal Günlüğü",
        "🚀 Backtest & Sermaye Eğrisi",
        "🧠 Model & Öznitelik Önem Düzeyleri",
        "📅 Canlı Takip & İleriye Dönük Sermaye",
    ])

    # SEKME 1: SQLite Sinyal Logları
    with tab1:
        st.markdown("### SQLite 'signals' Tablosu Canlı Kayıtları")
        db_signals = signal_mgr.get_signals(ticker=selected_ticker, limit=40)
        
        if not db_signals.empty:
            c_tot, c_buy, c_sell, c_hold = st.columns(4)
            c_tot.metric("Toplam Kayıtlı Sinyal", f"{len(db_signals)} adet")
            c_buy.metric("AL Sinyalleri", f"{(db_signals['signal_label'] == 'AL').sum()} adet")
            c_sell.metric("SAT Sinyalleri", f"{(db_signals['signal_label'] == 'SAT').sum()} adet")
            c_hold.metric("NÖTR Sinyalleri", f"{(db_signals['signal_label'] == 'NOTR').sum()} adet")
            
            st.markdown("<br>", unsafe_allow_html=True)
            display_cols = [
                "id", "timestamp", "signal_label", "confidence",
                "entry_price", "stop_loss", "take_profit", "status", "pnl_pct"
            ]
            
            st.dataframe(
                db_signals[display_cols],
                column_config={
                    "id": st.column_config.NumberColumn("ID", format="%d"),
                    "timestamp": st.column_config.TextColumn("Zaman Damgası"),
                    "signal_label": st.column_config.TextColumn("Tahmin"),
                    "confidence": st.column_config.ProgressColumn("Güven", min_value=0.0, max_value=1.0, format="%.1f%%"),
                    "entry_price": st.column_config.NumberColumn("Giriş Fiyatı", format="$%.2f"),
                    "stop_loss": st.column_config.NumberColumn("Stop-Loss", format="$%.2f"),
                    "take_profit": st.column_config.NumberColumn("Kâr Al Hedefi", format="$%.2f"),
                    "status": st.column_config.TextColumn("Pozisyon Durumu"),
                    "pnl_pct": st.column_config.NumberColumn("PnL (%)", format="%+.2f%%"),
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("Henüz veritabanında kayıtlı sinyal bulunmuyor.")

    # SEKME 2: Backtest & Sermaye Büyümesi
    with tab2:
        st.markdown("### Geriye Dönük Çubuk Bazlı Simülasyon (Backtest)")
        backtester = BacktestEngine(signal_manager=signal_mgr, max_holding_bars=10, spot_only=True)
        
        with st.spinner("Backtest koşturuluyor (Spot Modeli)..."):
            preds_encoded = bundle["model"].predict(df_features[bundle["feature_names"]])
            preds_raw = pd.Series(preds_encoded).map({0: -1, 1: 0, 2: 1}).values
            
            # Spot işlem modelinde sadece AL (1) için SL ve TP tanımlanır (Açığa satış/Short yok)
            sl_series = np.where(
                preds_raw == 1,
                df_features["Close"] - (df_features["ATR"] * atr_multiplier),
                np.nan
            )
            tp_series = np.where(
                preds_raw == 1,
                df_features["Close"] + (df_features["ATR"] * atr_multiplier * risk_reward),
                np.nan
            )

            bt_res = backtester.run_backtest(
                df_bars=df_features,
                signals_df=df_features.assign(
                    signal_raw=preds_raw,
                    stop_loss=sl_series,
                    take_profit=tp_series,
                    confidence=0.70,
                ),
                ticker=selected_ticker,
                save_to_db=False,
            )

        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Toplam İşlem Sayısı", f"{bt_res['total_trades']} adet")
        b2.metric("Kazanma Oranı (Win Rate)", f"%{bt_res['win_rate']:.1f}", delta=f"{bt_res['win_trades']} Kazanç / {bt_res['loss_trades']} Kayıp")
        b3.metric("Kâr Faktörü (Profit Factor)", f"{bt_res['profit_factor']:.2f}", delta="> 1.5 Hedef")
        b4.metric("Kümülatif Getiri", f"%{bt_res['total_return_pct']:+.2f}", delta=f"Max DD: -%{bt_res['max_drawdown_pct']:.2f}", delta_color="inverse")

        # Görsel Sermaye Büyüme Eğrisi (Equity Curve with Gradient Area)
        if bt_res.get("equity_curve"):
            st.markdown("#### 📈 Sermaye Büyüme Eğrisi (100 Bazlı Portföy)")
            eq_curve = bt_res["equity_curve"]

            # Eksi değerleri sıfırda taban al (sermaye 0'ın altına düşmesin)
            eq_clipped = [max(v, 0) for v in eq_curve]
            x_labels = list(range(1, len(eq_clipped) + 1))

            # Kâr / zarar bölgelerini renklendirmek için ayrı izler
            profit_y = [v if v >= 100 else None for v in eq_clipped]
            loss_y   = [v if v < 100  else None for v in eq_clipped]

            fig_eq = go.Figure()

            # Kâr bölgesi (yeşil alan)
            fig_eq.add_trace(go.Scatter(
                x=x_labels, y=profit_y,
                mode="lines",
                line=dict(color="#00e676", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(0, 230, 118, 0.12)",
                name="Kâr Bölgesi",
                connectgaps=False,
            ))

            # Zarar bölgesi (kırmızı alan)
            fig_eq.add_trace(go.Scatter(
                x=x_labels, y=loss_y,
                mode="lines",
                line=dict(color="#ef5350", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(239, 83, 80, 0.12)",
                name="Zarar Bölgesi",
                connectgaps=False,
            ))

            # Başlangıç (100) referans çizgisi
            fig_eq.add_hline(
                y=100,
                line_dash="dash",
                line_color="rgba(255,255,255,0.25)",
                annotation_text="Başlangıç Sermayesi",
                annotation_position="top left",
                annotation_font_color="rgba(255,255,255,0.45)",
            )

            # Tepe ve dip noktaları işaretle
            max_val = max(eq_clipped)
            min_val = min(eq_clipped)
            max_idx = eq_clipped.index(max_val) + 1
            min_idx = eq_clipped.index(min_val) + 1

            fig_eq.add_annotation(x=max_idx, y=max_val,
                text=f"⬆ Zirve: {max_val:.1f}",
                showarrow=True, arrowhead=2, arrowcolor="#00e676",
                font=dict(color="#00e676", size=12), bgcolor="#161b22",
                bordercolor="#00e676", borderwidth=1, borderpad=4,
            )
            if min_val < 100:
                fig_eq.add_annotation(x=min_idx, y=min_val,
                    text=f"⬇ Dip: {min_val:.1f}",
                    showarrow=True, arrowhead=2, arrowcolor="#ef5350",
                    font=dict(color="#ef5350", size=12), bgcolor="#161b22",
                    bordercolor="#ef5350", borderwidth=1, borderpad=4,
                )

            fig_eq.update_layout(
                template="plotly_dark",
                height=420,
                plot_bgcolor="#161b22",
                paper_bgcolor="#161b22",
                margin=dict(l=60, r=30, t=40, b=60),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    font=dict(size=13),
                ),
                yaxis=dict(
                    title="Portföy Değeri (Başlangıç = 100)",
                    title_font=dict(size=13),
                    tickfont=dict(size=12),
                    gridcolor="rgba(255,255,255,0.07)",
                    rangemode="tozero",   # Eksi kısım hiç gösterilmez
                ),
                xaxis=dict(
                    title="İşlem Sırası",
                    title_font=dict(size=13),
                    tickfont=dict(size=12),
                    gridcolor="rgba(255,255,255,0.07)",
                    fixedrange=True,      # Geriye kaydırma / pan devre dışı
                    rangeslider=dict(visible=False),
                ),
                hovermode="x unified",
            )
            st.plotly_chart(fig_eq, use_container_width=True, config={
                "displayModeBar": False,   # Üst araç çubuğunu gizle (zoom/pan butonları)
                "scrollZoom": False,       # Scroll ile zoom devre dışı
            })

    # SEKME 3: Model & Öznitelik Önem Düzeyleri
    with tab3:
        st.markdown("### Karar Mekanizmasını Yöneten En Önemli İndikatörler")
        trainer = FinancialModelTrainer()
        imp_df = trainer.get_feature_importance(bundle["model"], bundle["feature_names"], top_n=12)

        # Plotly Yatay Çubuk Grafiği
        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"] * 100,
            y=imp_df["Feature"],
            orientation="h",
            marker=dict(
                color=imp_df["Importance"],
                colorscale="Viridis",
            ),
        ))
        fig_imp.update_layout(
            template="plotly_dark",
            height=380,
            plot_bgcolor="#161b22",
            paper_bgcolor="#161b22",
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Önem Yüzdesi (%)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # --------------------------------------------------------------------------
    # SEKME 4: Canlı Takip & İleriye Dönük Sermaye Büyümesi
    # --------------------------------------------------------------------------
    with tab4:
        # Canlı Takip Tarih & Sermaye Kontrolleri
        col_ctrl1, col_ctrl2 = st.columns([2, 1])
        with col_ctrl1:
            min_date = datetime(2026, 11, 1).date()
            live_start_picker = st.date_input(
                "📅 Takip Başlangıç Tarihi (Spot Portföy)",
                value=min_date,
                min_value=min_date,
                help="Sermaye takibinin başlayacağı tarih (1 Kasım 2026 öncesine alınamaz)",
            )
            LIVE_START_DATE = str(live_start_picker)
        with col_ctrl2:
            initial_cap_input = st.number_input(
                "💵 Başlangıç Sermayesi ($)",
                value=1000.0,
                step=100.0,
                min_value=50.0,
                help="Portföy simülasyonu başlangıç nakit sermayesi",
            )
            INITIAL_CAPITAL = float(initial_cap_input)

        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #161b22 0%, #1a2332 100%);
                        border: 1px solid #30363d; border-radius: 10px;
                        padding: 16px 20px; margin-bottom: 18px;">
                <span style="font-size: 0.8rem; color: #58a6ff; font-weight: 700; letter-spacing: 0.08em;">
                    ⚡ CANLI SPOT TAKİP VE SERMAYE YÖNETİMİ
                </span>
                <h3 style="color: #f0f6fc; margin: 4px 0; font-size: 1.4rem;">
                    {LIVE_START_DATE} — Başlangıç Sermayesi: <span style="color: #26a69a;">${INITIAL_CAPITAL:,.0f}</span>
                </h3>
                <span style="color: #8b949e; font-size: 0.85rem;">
                    Model yalnızca <b>Spot Alış</b> ve nakit yönetimi yapar (Kaldıraçsız / Short pozisyon açılmaz).
                    Açık spot pozisyon kâr hedefine (TP) veya zarar durdurma (SL) seviyesine ulaştığında kapatılır.
                </span>
            </div>
        """, unsafe_allow_html=True)

        # Açık pozisyonları güncelle (SL/TP kontrolü)
        with st.spinner("Açık pozisyonlar güncelleniyor..."):
            updated = signal_mgr.resolve_open_positions(
                ticker=selected_ticker,
                df_bars=df_features,
            )
            if updated > 0:
                st.success(f"✅ {updated} pozisyon kapatıldı/güncellendi.")

        # Başlangıç tarihinden itibaren sinyalleri çek
        live_df = signal_mgr.get_live_signals(ticker=selected_ticker, start_date=LIVE_START_DATE)

        if live_df.empty:
            st.info("📭 Henüz bu tarihten itibaren kayıtlı sinyal bulunmuyor. "
                    "Sistem her sayfayı yükleyişte bugünkü sinyali otomatik kaydeder.")
        else:
            # --- Özet KPI Kartları ---
            closed_df = live_df[live_df["status"].isin(["HIT_TP", "HIT_SL", "EXPIRED"])]
            open_df = live_df[live_df["status"] == "OPEN"]
            win_trades = (closed_df["pnl_pct"] > 0).sum() if not closed_df.empty else 0
            loss_trades = (closed_df["pnl_pct"] <= 0).sum() if not closed_df.empty else 0
            win_rate = (win_trades / len(closed_df) * 100) if len(closed_df) > 0 else 0.0

            # Gerçekleşmiş Nakit Kasa Büyümesi
            realized_equity = INITIAL_CAPITAL
            eq_points = [{"tarih": LIVE_START_DATE, "deger": realized_equity}]
            for _, row in closed_df.sort_values("exit_timestamp").iterrows():
                if pd.notna(row["pnl_pct"]):
                    realized_equity = realized_equity * (1 + float(row["pnl_pct"]) / 100)
                    eq_points.append({
                        "tarih": str(row["exit_timestamp"])[:10],
                        "deger": round(realized_equity, 2),
                    })

            # Açık pozisyonların anlık gerçekleşmemiş (yüzen) kâr/zararı
            unrealized_pnl_usd = 0.0
            unrealized_pnl_pct = 0.0

            if not open_df.empty:
                open_df = open_df.copy()

                def _calc_open_metrics(row):
                    ep = float(row["entry_price"]) if pd.notna(row["entry_price"]) else last_close
                    sig = str(row["signal_label"])
                    sl = float(row["stop_loss"]) if pd.notna(row["stop_loss"]) else None
                    tp = float(row["take_profit"]) if pd.notna(row["take_profit"]) else None

                    if sig == "AL":
                        pnl = ((last_close - ep) / ep) * 100
                        dist_tp = ((tp - last_close) / last_close) * 100 if tp else 0.0
                        dist_sl = ((last_close - sl) / last_close) * 100 if sl else 0.0
                    else:
                        pnl = ((ep - last_close) / ep) * 100
                        dist_tp = ((last_close - tp) / last_close) * 100 if tp else 0.0
                        dist_sl = ((sl - last_close) / last_close) * 100 if sl else 0.0

                    return pd.Series([last_close, round(pnl, 2), round(dist_tp, 2), round(dist_sl, 2)])

                open_df[["current_price", "unrealized_pnl_pct", "dist_to_tp", "dist_to_sl"]] = open_df.apply(_calc_open_metrics, axis=1)

                latest_open = open_df.iloc[-1]
                unrealized_pnl_pct = float(latest_open["unrealized_pnl_pct"])
                unrealized_pnl_usd = realized_equity * (unrealized_pnl_pct / 100)

            # Anlık Toplam Özkaynak (Floating Equity = Kasa + Anlık Yüzen K/Z)
            floating_equity = realized_equity + unrealized_pnl_usd
            total_return_pct = ((floating_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

            # KPI Satırı
            lk1, lk2, lk3, lk4, lk5 = st.columns(5)
            eq_color = "#26a69a" if floating_equity >= INITIAL_CAPITAL else "#ef5350"
            eq_icon = "▲" if floating_equity >= INITIAL_CAPITAL else "▼"

            cash_sub = f"Kasa: ${realized_equity:,.2f}"
            if len(open_df) > 0:
                pnl_sign = "+" if unrealized_pnl_usd >= 0 else ""
                cash_sub += f" | Açık: {pnl_sign}${unrealized_pnl_usd:,.2f}"

            with lk1:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Anlık Portföy (Özkaynak)</div>
                        <div class="kpi-value" style="color: {eq_color};">${floating_equity:,.2f}</div>
                        <div class="kpi-sub" style="color: {eq_color};">{eq_icon} %{total_return_pct:+.2f}</div>
                        <div style="font-size: 0.72rem; color: #8b949e; margin-top: 4px;">{cash_sub}</div>
                    </div>
                """, unsafe_allow_html=True)

            with lk2:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Kapatılan İşlem</div>
                        <div class="kpi-value">{len(closed_df)}</div>
                        <div class="kpi-sub" style="color: #8b949e;">{win_trades} Kâr / {loss_trades} Zarar</div>
                    </div>
                """, unsafe_allow_html=True)

            with lk3:
                wr_color = "#26a69a" if win_rate >= 50 else "#ef5350"
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Kazanma Oranı</div>
                        <div class="kpi-value" style="color: {wr_color};">%{win_rate:.1f}</div>
                        <div class="kpi-sub" style="color: #8b949e;">Kapalı pozisyonlar üzerinden</div>
                    </div>
                """, unsafe_allow_html=True)

            with lk4:
                avg_pnl = closed_df["pnl_pct"].mean() if not closed_df.empty else 0.0
                ap_color = "#26a69a" if avg_pnl >= 0 else "#ef5350"
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Ortalama İşlem PnL</div>
                        <div class="kpi-value" style="color: {ap_color};">%{avg_pnl:+.2f}</div>
                        <div class="kpi-sub" style="color: #8b949e;">İşlem başına ortalama</div>
                    </div>
                """, unsafe_allow_html=True)

            with lk5:
                st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Açık Pozisyon</div>
                        <div class="kpi-value" style="color: #58a6ff;">{len(open_df)}</div>
                        <div class="kpi-sub" style="color: #8b949e;">Bekleyen (SL/TP aktif)</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

            # --- İleriye Dönük Sermaye Eğrisi ---
            st.markdown("#### 📈 İleriye Dönük Sermaye Büyümesi (Gerçek Tarihli)")

            if len(eq_points) < 2:
                st.info("📊 Sermaye eğrisi için en az bir kapalı pozisyon gerekiyor. "
                        "İlk pozisyon kapandığında grafik şekillenecek.")
            else:
                eq_df = pd.DataFrame(eq_points)
                eq_df["tarih"] = pd.to_datetime(eq_df["tarih"])

                profit_y = [v if v >= INITIAL_CAPITAL else None for v in eq_df["deger"]]
                loss_y   = [v if v < INITIAL_CAPITAL else None for v in eq_df["deger"]]

                fig_live = go.Figure()

                # Kâr bölgesi (yeşil)
                fig_live.add_trace(go.Scatter(
                    x=eq_df["tarih"], y=profit_y,
                    mode="lines+markers",
                    line=dict(color="#00e676", width=2.5),
                    marker=dict(size=7, color="#00e676", line=dict(width=1.5, color="#fff")),
                    fill="tozeroy",
                    fillcolor="rgba(0, 230, 118, 0.10)",
                    name="Kâr Bölgesi ($)",
                    connectgaps=False,
                    hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Portföy: $%{y:,.2f}<extra></extra>",
                ))

                # Zarar bölgesi (kırmızı)
                fig_live.add_trace(go.Scatter(
                    x=eq_df["tarih"], y=loss_y,
                    mode="lines+markers",
                    line=dict(color="#ef5350", width=2.5),
                    marker=dict(size=7, color="#ef5350", line=dict(width=1.5, color="#fff")),
                    fill="tozeroy",
                    fillcolor="rgba(239, 83, 80, 0.10)",
                    name="Zarar Bölgesi ($)",
                    connectgaps=False,
                    hovertemplate="<b>%{x|%d.%m.%Y}</b><br>Portföy: $%{y:,.2f}<extra></extra>",
                ))

                # Başlangıç sermayesi referans çizgisi
                fig_live.add_hline(
                    y=INITIAL_CAPITAL,
                    line_dash="dash",
                    line_color="rgba(255,255,255,0.28)",
                    annotation_text=f"Başlangıç Sermayesi: ${INITIAL_CAPITAL:,.0f}",
                    annotation_position="top left",
                    annotation_font_color="rgba(255,255,255,0.5)",
                )

                # Zirve ve dip anotasyonları
                max_val = max(v for v in eq_df["deger"])
                min_val = min(v for v in eq_df["deger"])
                max_date = eq_df.loc[eq_df["deger"].idxmax(), "tarih"]
                min_date = eq_df.loc[eq_df["deger"].idxmin(), "tarih"]

                fig_live.add_annotation(
                    x=max_date, y=max_val,
                    text=f"⬆ Zirve: ${max_val:,.2f}",
                    showarrow=True, arrowhead=2, arrowcolor="#00e676",
                    font=dict(color="#00e676", size=12), bgcolor="#161b22",
                    bordercolor="#00e676", borderwidth=1, borderpad=4,
                )
                if min_val < INITIAL_CAPITAL:
                    fig_live.add_annotation(
                        x=min_date, y=min_val,
                        text=f"⬇ Dip: ${min_val:,.2f}",
                        showarrow=True, arrowhead=2, arrowcolor="#ef5350",
                        font=dict(color="#ef5350", size=12), bgcolor="#161b22",
                        bordercolor="#ef5350", borderwidth=1, borderpad=4,
                    )

                fig_live.update_layout(
                    template="plotly_dark",
                    height=420,
                    plot_bgcolor="#161b22",
                    paper_bgcolor="#161b22",
                    margin=dict(l=60, r=30, t=40, b=60),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=13)),
                    yaxis=dict(
                        title="Portföy Değeri (USD $)",
                        title_font=dict(size=13),
                        tickfont=dict(size=12),
                        tickprefix="$",
                        gridcolor="rgba(255,255,255,0.07)",
                        rangemode="tozero",
                    ),
                    xaxis=dict(
                        title="Tarih",
                        title_font=dict(size=13),
                        tickfont=dict(size=12),
                        tickformat="%d.%m.%Y",
                        gridcolor="rgba(255,255,255,0.07)",
                        rangeslider=dict(visible=False),
                    ),
                    hovermode="x unified",
                )
                st.plotly_chart(fig_live, use_container_width=True)

            st.markdown("<div style='height: 6px;'></div>", unsafe_allow_html=True)

            # --- Açık Pozisyonlar Tablosu ---
            if not open_df.empty:
                st.markdown("#### ⏳ Açık Pozisyonlar (SL / TP Bekliyor)")
                st.info(
                    f"ℹ️ **Pozisyon Halen Açık:** Fiyat henüz Kâr Al Hedefine (TP) veya Zarar Durdurma Seviyesine (SL) "
                    f"ulaşmadığı için işlem henüz nakite dönmemiştir. Tabloda güncel fiyat (${last_close:,.2f}) üzerinden hesaplanan "
                    f"**anlık gerçekleşmemiş kâr/zarar** ve hedeflere kalan mesafeler anlık izlenmektedir."
                )

                open_display = open_df[[
                    "id", "timestamp", "signal_label", "confidence",
                    "entry_price", "current_price", "unrealized_pnl_pct",
                    "take_profit", "dist_to_tp",
                    "stop_loss", "dist_to_sl"
                ]].copy()

                st.dataframe(
                    open_display,
                    column_config={
                        "id": st.column_config.NumberColumn("ID", format="%d"),
                        "timestamp": st.column_config.TextColumn("Giriş Tarihi"),
                        "signal_label": st.column_config.TextColumn("Sinyal"),
                        "confidence": st.column_config.ProgressColumn("Güven", min_value=0.0, max_value=1.0, format="%.1f%%"),
                        "entry_price": st.column_config.NumberColumn("Giriş", format="$%.2f"),
                        "current_price": st.column_config.NumberColumn("Güncel Fiyat", format="$%.2f"),
                        "unrealized_pnl_pct": st.column_config.NumberColumn("Anlık PnL", format="%+.2f%%"),
                        "take_profit": st.column_config.NumberColumn("Kâr Al Hedefi (TP)", format="$%.2f"),
                        "dist_to_tp": st.column_config.NumberColumn("TP'ye Kalan", format="%+.2f%%"),
                        "stop_loss": st.column_config.NumberColumn("Stop-Loss (SL)", format="$%.2f"),
                        "dist_to_sl": st.column_config.NumberColumn("SL'e Kalan", format="%+.2f%%"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )

            # --- Kapatılan Pozisyonlar Tablosu ---
            if not closed_df.empty:
                st.markdown("#### 📋 Kapatılan Pozisyonlar (Kâr/Zarar Kesinleşti)")
                closed_display = closed_df[[
                    "id", "timestamp", "signal_label", "entry_price",
                    "exit_price", "stop_loss", "take_profit", "status", "pnl_pct"
                ]].copy()

                def _status_emoji(s):
                    if s == "HIT_TP": return "✅ TP Vuruldu"
                    if s == "HIT_SL": return "❌ SL Vuruldu"
                    return "⏱️ Süresi Doldu"

                closed_display["status"] = closed_display["status"].apply(_status_emoji)

                st.dataframe(
                    closed_display,
                    column_config={
                        "id": st.column_config.NumberColumn("ID", format="%d"),
                        "timestamp": st.column_config.TextColumn("Giriş Tarihi"),
                        "signal_label": st.column_config.TextColumn("Sinyal"),
                        "entry_price": st.column_config.NumberColumn("Giriş", format="$%.2f"),
                        "exit_price": st.column_config.NumberColumn("Çıkış", format="$%.2f"),
                        "stop_loss": st.column_config.NumberColumn("SL", format="$%.2f"),
                        "take_profit": st.column_config.NumberColumn("TP", format="$%.2f"),
                        "status": st.column_config.TextColumn("Durum"),
                        "pnl_pct": st.column_config.NumberColumn("PnL (%)", format="%+.2f%%"),
                    },
                    hide_index=True,
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
