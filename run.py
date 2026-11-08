"""
Finansal Varlık Yön Tahmini ve Gerçek Zamanlı Sinyal Üretim Platformu
Merkezi Çalıştırıcı (Unified Runner)

Kullanım:
    python run.py                # Doğrudan Karar Destek Arayüzünü (Streamlit) açar
    python run.py --all          # 5 adımlı tam boru hattını çalıştırır (Test -> Faz 1 -> Faz 2 -> Faz 3 -> Arayüz)
    python run.py --train        # Sadece eğitim boru hattını çalıştırır (Faz 1-3)
    python run.py --test         # Sadece birim testleri çalıştırır
    python run.py --ticker ETH-USD  # Özel sembol belirler
"""

import argparse
import io
import os
import subprocess
import sys
import time
from pathlib import Path

# Windows UTF-8 desteği
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent


def print_banner():
    banner = r"""
======================================================================
  ⚡ FINANSAL TAHMIN VE KARAR DESTEK PLATFORMU
  Merkezi Yonetim & Calistirici (Unified Runner)
======================================================================
"""
    print(banner)


def run_command(cmd, step_name: str) -> bool:
    """Alt komutu çalıştırır ve sonucunu ekrana yansıtır."""
    print(f"\n[BASLIYOR] >>> {step_name}")
    print(f"Komut: {' '.join(cmd)}\n" + "-" * 60)
    start_time = time.time()
    
    try:
        res = subprocess.run(cmd, cwd=str(ROOT_DIR))
        elapsed = time.time() - start_time
        if res.returncode == 0:
            print("-" * 60)
            print(f"[BASARILI] {step_name} tamamlandi. ({elapsed:.1f} sn)\n")
            return True
        else:
            print("-" * 60)
            print(f"[HATA] {step_name} hata koduyla sonlandi: {res.returncode}\n")
            return False
    except KeyboardInterrupt:
        print(f"\n[IPTAL] Kullanici tarafindan durduruldu: {step_name}")
        return False
    except Exception as e:
        print(f"\n[HATA] Komut calistirilirken istisna olustu: {e}")
        return False


def run_tests() -> bool:
    cmd = [sys.executable, "-m", "unittest", "discover", "tests"]
    return run_command(cmd, "Adim 1: Birim Testleri Calistirma (Unit Tests)")


def run_phase1(ticker: str) -> bool:
    cmd = [sys.executable, "run_phase1.py", ticker]
    return run_command(cmd, f"Adim 2: Faz 1 - Veri & Oznitelik Muhendisligi ({ticker})")


def run_phase2(ticker: str) -> bool:
    cmd = [sys.executable, "run_phase2.py", ticker]
    return run_command(cmd, f"Adim 3: Faz 2 - Model Egitimi & Zaman Serisi CV ({ticker})")


def run_phase3(ticker: str) -> bool:
    cmd = [sys.executable, "run_phase3.py", ticker]
    return run_command(cmd, f"Adim 4: Faz 3 - SQLite Sinyal Loglama & Backtest ({ticker})")


def run_streamlit(port: int = 8501):
    print("\n" + "=" * 60)
    print(" 🚀 STREAMLIT KARAR DESTEK ARAYUZU BASLATILIYOR")
    print(f" Tarayici Adresi: http://localhost:{port}")
    print(" Durdurmak icin terminalde: Ctrl + C")
    print("=" * 60 + "\n")
    
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "app.py",
        f"--server.port={port}",
        "--server.headless=false",
    ]
    try:
        subprocess.run(cmd, cwd=str(ROOT_DIR))
    except KeyboardInterrupt:
        print("\n[BILGI] Arayuz sonlandirildi.")


def main():
    parser = argparse.ArgumentParser(
        description="Finansal Varlık Tahmin Platformu Tek Komut Çalıştırıcı",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornek Kullanimlar:
  python run.py                   # Arayuzu baslatir (en hizli kullanim)
  python run.py --all             # 5 adimi sirayla calistirir
  python run.py --train           # Faz 1, Faz 2 ve Faz 3'u calistirir
  python run.py --test            # Sadece testleri calistirir
  python run.py --ticker ETH-USD  # Farkli bir kripto/hisse ile calistirir
        """,
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="5 adimli tam boru hattini calistirir (Testler -> Faz 1 -> Faz 2 -> Faz 3 -> Streamlit)",
    )
    parser.add_argument(
        "--train", "-t",
        action="store_true",
        help="Arayuz acilmadan Faz 1, Faz 2 ve Faz 3 adimlarini sirayla calistirir",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Sadece birim testleri calistirir",
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="BTC-USD",
        help="Analiz edilecek varlik sembolu (Varsayilan: BTC-USD)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit port numarasi (Varsayilan: 8501)",
    )

    args = parser.parse_args()
    print_banner()

    # 1. Sadece test çalıştırma
    if args.test:
        success = run_tests()
        sys.exit(0 if success else 1)

    # 2. Sadece eğitim çalıştırma (Faz 1-3)
    if args.train:
        print(f"[*] Egitim ve test hatti '{args.ticker}' icin baslatiliyor...\n")
        if not run_phase1(args.ticker):
            sys.exit(1)
        if not run_phase2(args.ticker):
            sys.exit(1)
        if not run_phase3(args.ticker):
            sys.exit(1)
        print("\n[+] Egitim hatti basariyla tamamlandi!")
        sys.exit(0)

    # 3. Tüm boru hattını sırayla çalıştırma (5 adım)
    if args.all:
        print(f"[*] 5 Adimli Tam Boru Hatti '{args.ticker}' icin baslatiliyor...\n")
        if not run_tests():
            print("[!] Birim testler basarisiz oldu. Boru hatti durduruldu.")
            sys.exit(1)
        if not run_phase1(args.ticker):
            print("[!] Faz 1 basarisiz oldu.")
            sys.exit(1)
        if not run_phase2(args.ticker):
            print("[!] Faz 2 basarisiz oldu.")
            sys.exit(1)
        if not run_phase3(args.ticker):
            print("[!] Faz 3 basarisiz oldu.")
            sys.exit(1)
        
        print("\n[+] Tum on hazirlik adimlari basariyla gecti. Arayuz baslatiliyor...")
        run_streamlit(port=args.port)
        return

    # 4. Varsayılan (Tek Komut): Doğrudan Streamlit arayüzünü aç
    # Not: app.py zaten eksik modelleri otomatik olarak egitmektedir.
    run_streamlit(port=args.port)


if __name__ == "__main__":
    main()
