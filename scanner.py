"""
Dusus Tarayicisi (MA7 / MA25 / MA99 sikisma + kirilim)

Kural:
 1) Once yerlesmis bir yukselis TRENDI olmali (MA7 > MA25 > MA99, MA99 dusmuyor)
 2) MA7, MA25'i asagi kesmis olmali
 3) Fiyat uc MA'nin olusturdugu bolgeye girmis olmali
 4) Fiyat bu bolgenin ALTINA kapanisla kirmis olmali (taze kirilim)
 5) Uc MA birbirine ne kadar yakinsa (spread %) o kadar ustte listelenir

Kullanim:
    python scanner.py
    python scanner.py --interval 1h --top 150 --max-spread 2
Veri: Binance herkese acik piyasa verisi (anahtar gerekmez). Yapay zeka KULLANMAZ.
Bu bir yatirim tavsiyesi degildir, sadece bir tarama aracidir.
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

BASE_URLS = [
    "https://api.binance.com",
    "https://data-api.binance.vision",
    "https://api.binance.us",
]
BAD_SUFFIXES = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
STABLES = {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "EURUSDT", "DAIUSDT", "USDEUSDT"}
LOG_FILE = "signals_log.csv"


def compute_bands(df):
    """MA7/25/99 ve bolge sinirlarini bir kez hesaplar (hizli geriye donuk test icin)."""
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    s = pd.Series(close)
    ma7 = s.rolling(7).mean().to_numpy()
    ma25 = s.rolling(25).mean().to_numpy()
    ma99 = s.rolling(99).mean().to_numpy()
    stack = np.vstack([ma7, ma25, ma99])
    return {"close": close, "high": high, "low": low, "ma7": ma7, "ma25": ma25, "ma99": ma99,
            "lo": stack.min(axis=0), "hi": stack.max(axis=0)}


def signal_at(b, last, uptrend_pct=5.0, uptrend_lookback=48, cross_within=24,
              break_fresh=3, max_spread_pct=3.0):
    """'last' indeksli mum itibariyle sinyal var mi? Varsa dict, yoksa None."""
    close, high, low = b["close"], b["high"], b["low"]
    ma7, ma25, ma99, lo, hi = b["ma7"], b["ma25"], b["ma99"], b["lo"], b["hi"]
    first_c = max(last - cross_within, 100)

    # En yeni kesisimden geriye dogru ara
    for c in range(last, first_c - 1, -1):
        if not (ma7[c - 1] >= ma25[c - 1] and ma7[c] < ma25[c]):
            continue

        # 0) Yukselis TRENDI yerlesmis olmali: kesisimden hemen once MA25 > MA99
        #    ve MA99 dusmuyor (dusus trendindeki kucuk sicramalari eler)
        if not (ma25[c - 1] > ma99[c - 1] and ma99[c - 1] >= ma99[max(c - 25, 0)]):
            continue

        # 1) Kesisimden onceki yukselis
        start = max(c - uptrend_lookback, 0)
        window = close[start:c + 1]
        rise = (window.max() / window[0] - 1) * 100
        if rise < uptrend_pct:
            continue

        # 3) bolgeye giris, 4) bolgenin altina kirilim
        entered = None
        broke = None
        for k in range(c, last + 1):
            if entered is None and low[k] <= hi[k] and high[k] >= lo[k]:
                entered = k
                continue
            if entered is not None and close[k] < lo[k]:
                broke = k
                break
        if broke is None or (last - broke) >= break_fresh:
            continue

        # 5) sikisma (yakinlik)
        spread = (hi[broke] - lo[broke]) / close[broke] * 100
        if spread > max_spread_pct:
            continue
        return {
            "break_index": broke,
            "cross_index": c,
            "rise_pct": round(float(rise), 2),
            "spread_pct": round(float(spread), 2),
            "price": float(close[broke]),
            "last_price": float(close[last]),
        }
    return None


def signal_down_at(b, last, max_spread_pct=3.0, zone_within=12):
    """DUSUS TRENDI modu: yerlesmis dusus trendinde MA'lar sikisir, fiyat bandin icine girer
    ve bandin ALTINA ilk kez kapanisla kirar."""
    close, high, low = b["close"], b["high"], b["low"]
    ma25, ma99, lo, hi = b["ma25"], b["ma99"], b["lo"], b["hi"]
    k = last
    if k < 125:
        return None
    if not (close[k] < lo[k]):                    # bandin altinda kapanis
        return None
    if not (close[k - 1] >= lo[k - 1]):           # taze kirilim
        return None
    if not any(low[j] <= hi[j] and high[j] >= lo[j] for j in range(k - zone_within, k)):
        return None                               # son mumlarda fiyat bandin icindeydi
    if not (ma25[k] < ma99[k] and ma99[k] <= ma99[k - 25]):
        return None                               # yerlesmis dusus trendi
    spread = (hi[k] - lo[k]) / close[k] * 100
    if spread > max_spread_pct:
        return None
    return {"break_index": k, "cross_index": k, "rise_pct": 0.0,
            "spread_pct": round(float(spread), 2), "price": float(close[k]),
            "last_price": float(close[last])}


def detect_down(df, max_spread_pct=3.0, break_fresh=3, zone_within=12):
    if len(df) < 130:
        return None
    b = compute_bands(df)
    last = len(df) - 1
    for k in range(last, last - break_fresh, -1):
        s = signal_down_at(b, k, max_spread_pct, zone_within)
        if s:
            s["last_price"] = float(b["close"][last])
            return s
    return None


def detect(df, uptrend_pct=5.0, uptrend_lookback=48, cross_within=24,
           break_fresh=3, max_spread_pct=3.0):
    """df: sadece KAPANMIS mumlar. Sinyal varsa dict, yoksa None."""
    if len(df) < 130:
        return None
    b = compute_bands(df)
    return signal_at(b, len(df) - 1, uptrend_pct, uptrend_lookback, cross_within,
                     break_fresh, max_spread_pct)


def get_json(path, params=None):
    last_err = None
    for base in BASE_URLS:
        try:
            r = requests.get(base + path, params=params, timeout=15)
            if r.status_code == 200:
                return r.json(), base
            last_err = f"{base} -> HTTP {r.status_code}"
        except Exception as e:  # noqa
            last_err = f"{base} -> {e}"
    raise RuntimeError("Binance verisine ulasilamadi. Son hata: " + str(last_err))


def klines(symbol, interval, limit=300):
    data, _ = get_json("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "qvol", "trades", "tb", "tq", "ig"])
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.iloc[:-1].reset_index(drop=True)  # son (kapanmamis) mumu at


def top_symbols(n):
    tickers, base = get_json("/api/v3/ticker/24hr")
    rows = []
    for t in tickers:
        s = t["symbol"]
        if not s.endswith("USDT") or s in STABLES or s.endswith(BAD_SUFFIXES):
            continue
        rows.append((s, float(t["quoteVolume"]), float(t["priceChangePercent"])))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:n], base

def send_telegram(message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")

    if not token or not chat_id:
        print("Telegram bilgileri bulunamadi; bildirim gonderilmedi.")
        return

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": message},
            timeout=15,
        )
        response.raise_for_status()
        print("Telegram bildirimi gonderildi.")
    except Exception as e:
        print(f"Telegram bildirimi gonderilemedi: {e}")

def log_signal(row):
    exists = os.path.exists(LOG_FILE)
    seen = set()
    if exists:
        with open(LOG_FILE, newline="") as f:
            for r in csv.DictReader(f):
                seen.add((r["symbol"], r["interval"], r["break_time_utc"]))
    key = (row["symbol"], row["interval"], row["break_time_utc"])
    if key in seen:
        return
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="30m", help="15m, 30m, 1h, 4h, 1d")
    p.add_argument("--top", type=int, default=100, help="hacme gore ilk kac coin")
    p.add_argument("--max-spread", type=float, default=3.0, help="MA'lar arasi azami fark (%%)")
    p.add_argument("--uptrend", type=float, default=5.0, help="onceki yukselis (%%)")
    p.add_argument("--lookback", type=int, default=48, help="yukselis icin bakilan mum sayisi")
    p.add_argument("--cross-within", type=int, default=24, help="kesisim en fazla kac mum once")
    p.add_argument("--fresh", type=int, default=3, help="kirilim en fazla kac mum once")
    p.add_argument("--mode", choices=["cross", "down"], default="cross",
                   help="cross: yukselis sonrasi MA kesisimi (varsayilan) | down: yerlesmis dusus trendinde sikisma kirilimi")
    a = p.parse_args()

    try:
        syms, base = top_symbols(a.top)
    except Exception as e:
        print("HATA:", e)
        print("Not: Codespaces ABD'de calistigi icin Binance engelleyebilir. Bu mesaji bana yaz.")
        sys.exit(1)

    print(f"Kaynak: {base} | Zaman dilimi: {a.interval} | Taranan coin: {len(syms)}")
    found = []
    for i, (s, qv, chg) in enumerate(syms, 1):
        try:
            df = klines(s, a.interval)
            if a.mode == "down":
                res = detect_down(df, a.max_spread, a.fresh)
            else:
                res = detect(df, a.uptrend, a.lookback, a.cross_within, a.fresh, a.max_spread)
        except Exception as e:  # tek coin hata verirse devam et
            print(f"  {s}: atlandi ({e})")
            continue
        if res:
            bt = df["open_time"].iloc[res["break_index"]]
            res.update({"symbol": s, "chg24h": chg, "qvol_musd": round(qv / 1e6, 1), "break_time": bt})
            found.append(res)
        time.sleep(0.1)

    found.sort(key=lambda r: r["spread_pct"])
    print("\n=== DUSUS UYARISI VEREN COINLER (en sikisik en ustte) ===")
    if not found:
        print("Su an kurala uyan coin yok. (Esikleri gevsetmek icin --max-spread 5 deneyebilirsin.)")
    for r in found:
        bt_tr = r["break_time"].tz_convert("Europe/Istanbul").strftime("%d.%m %H:%M")
        print(f"{r['symbol']:<12} sikisma %{r['spread_pct']:<5} {('dusus trendi' if a.mode == 'down' else 'onceki yukselis %' + str(r['rise_pct'])):<22} "
              f"kirilim {bt_tr} (TR)  fiyat {r['last_price']}  24s %{r['chg24h']}")
        log_signal({
            "scan_time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "symbol": r["symbol"], "interval": a.interval,
            "break_time_utc": r["break_time"].strftime("%Y-%m-%d %H:%M"),
            "price_at_break": r["price"], "spread_pct": r["spread_pct"],
            "rise_pct": r["rise_pct"],
        })
    if found:
        print(f"\nSinyaller {LOG_FILE} dosyasina da kaydedildi (sonradan basari olcmek icin).")
        message = "🔔 COIN TARAMA SONUÇLARI\n\n"
        for r in found:
            message += (
                f"Coin: {r['symbol']}\n"
                f"Sıkışma: %{r['spread_pct']}\n"
                f"Fiyat: {r['last_price']}\n"
                f"24 saatlik değişim: %{r['chg24h']}\n\n"
            )
        send_telegram(message)

if __name__ == "__main__":
    main()