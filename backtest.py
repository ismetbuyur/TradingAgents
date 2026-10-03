"""
Geriye Donuk Test (Backtest): scanner.py'deki kural gecmiste ise yaramis mi?

Her sinyal icin "dusus" yonunde (short) su sorulur:
  Sinyal mumunun kapanisindan sonra, --horizon mum icinde fiyat once
  %tp asagi mi gitti (KAZANC), yoksa once %sl yukari mi gitti (KAYIP)?
  Ikisi de olmazsa sure dolunca kapanis farki alinir.
Karsilastirma icin ayni olcum RASTGELE mumlarda da yapilir (baz oran).
Sinyaller sikisma (spread) araligina ve DELTA (emir akisi) teyidine gore gruplanir ve veri ikiye bolunur
(ilk yari / ikinci yari) ki tesadufe uyum olup olmadigi gorulsun.

Kullanim:
    python backtest.py
    python backtest.py --days 90 --top 80 --tp 3 --sl 3 --horizon 48
Yatirim tavsiyesi degildir. Gecmis sonuc gelecegi garanti etmez.
"""
import argparse
import csv
import sys
import time

import numpy as np
import pandas as pd

from scanner import compute_bands, get_json, signal_at, signal_down_at, top_symbols

MS = {"15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}


def to_df(rows):
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "qvol", "trades", "tb", "tq", "ig"])
    for col in ("open", "high", "low", "close", "volume", "tb"):
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def fetch_history(symbol, interval, days):
    need = int(days * 24 * 60 / MS[interval])
    rows, end = [], None
    while len(rows) < need + 1:
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        if end:
            params["endTime"] = end
        data, _ = get_json("/api/v3/klines", params)
        if not data:
            break
        rows = data + rows
        end = data[0][0] - 1
        if len(data) < 1000:
            break
        time.sleep(0.1)
    df = to_df(rows).iloc[:-1]            # kapanmamis son mumu at
    return df.tail(need).reset_index(drop=True)


def outcome(b, t, horizon, tp, sl):
    """Short mantigi: fiyat once tp% duserse kazanc, once sl% yukselirse kayip."""
    close, high, low = b["close"], b["high"], b["low"]
    if t + horizon >= len(close):
        return None
    entry = close[t]
    up = entry * (1 + sl / 100)
    dn = entry * (1 - tp / 100)
    for j in range(t + 1, t + horizon + 1):
        if high[j] >= up:               # ayni mumda ikisi de olursa kayip say (temkinli)
            return -sl, "SL"
        if low[j] <= dn:
            return tp, "TP"
    return (entry - close[t + horizon]) / entry * 100, "SURE"


def run_symbol(df, a):
    b = compute_bands(df)
    n = len(df)
    # ATR(14): ortalama mum oynakligi
    prev = np.roll(b["close"], 1)
    tr = np.maximum(b["high"] - b["low"],
                    np.maximum(np.abs(b["high"] - prev), np.abs(b["low"] - prev)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    # Delta: agresif alim hacmi - agresif satim hacmi (mum bazinda, Binance 'taker buy' verisi)
    vol = df["volume"].to_numpy(float)
    dlt = 2 * df["tb"].to_numpy(float) - vol
    sigs, base = [], []
    next_ok = 0
    for t in range(130, n - a.horizon - 1):
        if t % 6 == 0:                  # baz oran icin rastgele (duzenli) ornekler
            r = outcome(b, t, a.horizon, a.tp, a.sl)
            if r:
                base.append({"ret": r[0] - a.fee, "label": r[1], "frac": t / n})
        if t < next_ok:
            continue
        if a.mode == "down":
            s = signal_down_at(b, t, a.max_spread)
        else:
            s = signal_at(b, t, a.uptrend, a.lookback, a.cross_within, 1, a.max_spread)
        if not s:
            continue
        r = outcome(b, t, a.horizon, a.tp, a.sl)
        if not r:
            continue

        # --- Hedef/durdurma yontemlerini ayni sinyal uzerinde karsilastir ---
        entry = b["close"][t]
        modes = {}

        def add(name, tp_pct, sl_pct):
            if tp_pct is None or sl_pct is None or tp_pct < 0.3 or sl_pct < 0.1:
                return
            o = outcome(b, t, a.horizon, tp_pct, sl_pct)
            if o:
                modes[name] = {"ret": o[0] - a.fee, "label": o[1], "tp": tp_pct, "sl": sl_pct}

        add("sabit", a.tp, a.sl)
        atr_pct = atr[t] / entry * 100
        if not np.isnan(atr_pct):
            add("atr", a.atr_mult * atr_pct, a.atr_mult * atr_pct)
        # Fibonacci: yukselisin dibi -> tepesi arasi geri cekilme seviyeleri
        ws = max(s["cross_index"] - 2 * a.lookback, 0)
        pk = ws + int(np.argmax(b["high"][ws:t + 1]))
        peak = b["high"][pk]
        swing_low = b["low"][ws:pk + 1].min()
        if a.mode != "down" and peak > swing_low and peak > entry:
            for name, ratio in (("fib38", 0.382), ("fib50", 0.5), ("fib62", 0.618)):
                tp_price = peak - ratio * (peak - swing_low)
                if tp_price < entry * 0.997:      # hedef giris fiyatinin yeterince altinda olmali
                    add(name, (entry - tp_price) / entry * 100, (peak / entry - 1) * 100)

        k = a.delta_window
        vsum = vol[t - k + 1:t + 1].sum()
        dr = float(dlt[t - k + 1:t + 1].sum() / vsum) if vsum > 0 else 0.0
        sigs.append({"time": df["open_time"][t], "spread": s["spread_pct"], "delta": dr,
                     "rise": s["rise_pct"], "ret": r[0] - a.fee,
                     "label": r[1], "frac": t / n, "modes": modes})
        next_ok = t + a.horizon         # ust uste binen islemleri sayma
    return sigs, base


def delta_report(sigs, k):
    print(f"\nDELTA TEYIDI (kirilim aninda son {k} mumda agresif alim-satim dengesi):")
    print("  delta < 0: satici baskin (dusus sinyaliyle ayni yon) | delta >= 0: alici baskin")
    neg = [r for r in sigs if r["delta"] < 0]
    pos = [r for r in sigs if r["delta"] >= 0]
    print("  delta <  0     :", stats(neg))
    print("  delta >= 0     :", stats(pos))
    srt = sorted(sigs, key=lambda r: r["delta"])
    third = len(srt) // 3
    if third >= 5:
        print("  En satici-baskin 1/3:", stats(srt[:third]))
        print("  Orta 1/3            :", stats(srt[third:2 * third]))
        print("  En alici-baskin 1/3 :", stats(srt[2 * third:]))
    if neg:
        print("  delta<0 grubu, ilk yari :", stats([r for r in neg if r["frac"] < 0.5]))
        print("  delta<0 grubu, ikinci yari:", stats([r for r in neg if r["frac"] >= 0.5]))
    print("  Okuma: delta<0 grubu delta>=0 grubundan belirgin iyiyse VE iki yaride de tutuyorsa filtre ise yariyor olabilir.")


def mode_report(sigs, fee):
    names = [("sabit", "Sabit % hedef / durdurma"),
             ("atr", "ATR (hedef = durdurma = k x ATR)"),
             ("fib38", "Fibonacci %38,2 hedef, durdurma = tepe"),
             ("fib50", "Fibonacci %50 hedef, durdurma = tepe"),
             ("fib62", "Fibonacci %61,8 hedef, durdurma = tepe")]
    print("\nHEDEF YONTEMLERI (ayni sinyaller, ayni 24 saatlik sure):")
    print("  'basabas' = bu hedef/durdurma oranlarinda zarar etmemek icin gereken kazanma yuzdesi")
    for key, label in names:
        rows = [x["modes"][key] for x in sigs if key in x["modes"]]
        if not rows:
            print(f"  {label:<42}: uygun sinyal yok")
            continue
        rets = np.array([r["ret"] for r in rows])
        tp = float(np.mean([r["tp"] for r in rows]))
        sl = float(np.mean([r["sl"] for r in rows]))
        be = (sl + fee) / (tp + sl) * 100
        win = (rets > 0).mean() * 100
        print(f"  {label:<42}: {len(rows):>4} islem | kazanma %{win:5.1f} (basabas %{be:4.1f}) | "
              f"ort. getiri %{rets.mean():+.2f} | ort. hedef %{tp:.1f} / durdurma %{sl:.1f}")
    print("  Not: 5 yontem denendi; birinin digerlerinden iyi cikmasi tesadufe bagli olabilir.")
    print("  Kazanma yuzdesi 'basabas'in belirgin ustundeyse ve iki yaride de tutarliysa ciddiye al.")


def stats(rows):
    if not rows:
        return "  0 islem"
    n = len(rows)
    rets = np.array([r["ret"] for r in rows])
    win = (rets > 0).mean() * 100
    lab = [r["label"] for r in rows]
    return (f"{n:>5} islem | kazanma %{win:5.1f} | ort. getiri %{rets.mean():+.2f} | "
            f"TP %{lab.count('TP')/n*100:.0f}  SL %{lab.count('SL')/n*100:.0f}  "
            f"SURE %{lab.count('SURE')/n*100:.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", default="30m", choices=list(MS))
    p.add_argument("--days", type=int, default=60, help="kac gunluk gecmis")
    p.add_argument("--top", type=int, default=50, help="hacme gore ilk kac coin")
    p.add_argument("--tp", type=float, default=3.0, help="hedef dusus (%%)")
    p.add_argument("--sl", type=float, default=3.0, help="zarar durdurma yukselis (%%)")
    p.add_argument("--horizon", type=int, default=48, help="en fazla kac mum beklenir")
    p.add_argument("--fee", type=float, default=0.2, help="toplam islem maliyeti (%%)")
    p.add_argument("--delta-window", type=int, default=6, help="delta icin bakilan mum sayisi")
    p.add_argument("--mode", choices=["cross", "down"], default="cross",
                   help="cross: yukselis sonrasi MA kesisimi | down: yerlesmis dusus trendinde sikisma kirilimi")
    p.add_argument("--atr-mult", type=float, default=2.0, help="ATR yonteminde carpan")
    p.add_argument("--max-spread", type=float, default=3.0)
    p.add_argument("--uptrend", type=float, default=5.0)
    p.add_argument("--lookback", type=int, default=48)
    p.add_argument("--cross-within", type=int, default=24)
    a = p.parse_args()

    syms, base_url = top_symbols(a.top)
    print(f"Mod: {a.mode} | Kaynak: {base_url} | {a.interval} | {a.days} gun | {len(syms)} coin | "
          f"hedef %{a.tp} / durdurma %{a.sl} / en fazla {a.horizon} mum | maliyet %{a.fee}")
    print("Veri indiriliyor, birkac dakika surebilir...\n")

    sigs, base = [], []
    for i, (s, _, _) in enumerate(syms, 1):
        try:
            df = fetch_history(s, a.interval, a.days)
            if len(df) < 300:
                continue
            ss, bb = run_symbol(df, a)
        except Exception as e:
            print(f"  {s}: atlandi ({e})")
            continue
        for x in ss:
            x["symbol"] = s
        sigs += ss
        base += bb
        if i % 10 == 0:
            print(f"  {i}/{len(syms)} coin islendi, su ana kadar {len(sigs)} sinyal")

    print("\n================ SONUC (dusus / short mantigi) ================")
    print("TUM SINYALLER :", stats(sigs))
    print("BAZ ORAN (rastgele mumlar):", stats(base))
    print("\nSikismaya gore (kucuk sayi = MA'lar birbirine yakin):")
    for lo, hi in [(0, 1), (1, 2), (2, 3), (3, 999)]:
        grp = [r for r in sigs if lo <= r["spread"] < hi]
        name = f"%{lo}-{hi}" if hi < 999 else f"%{lo}+"
        print(f"  sikisma {name:<7}:", stats(grp))
    print("\nVeriyi ikiye bol (tesadufe uyum kontrolu):")
    print("  ilk yari   :", stats([r for r in sigs if r["frac"] < 0.5]))
    print("  ikinci yari:", stats([r for r in sigs if r["frac"] >= 0.5]))
    delta_report(sigs, a.delta_window)
    mode_report(sigs, a.fee)
    if len(sigs) < 30:
        print("\nUYARI: 30'dan az sinyal var, sonuclar guvenilir degil. "
              "--days veya --top degerini artir ya da --max-spread gevset.")
    print("\nNasil okunur: sinyal sonuclari baz oranin belirgin sekilde ustunde degilse, "
          "kural tek basina avantaj saglamiyor demektir.")

    if sigs:
        with open("backtest_signals.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["symbol", "time", "spread", "delta", "rise", "ret", "label", "frac"])
            w.writeheader()
            w.writerows([{k: v for k, v in x.items() if k != "modes"} for x in sigs])
        print("Tum sinyaller backtest_signals.csv dosyasina yazildi.")


if __name__ == "__main__":
    main()