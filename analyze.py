"""
Hafif Analist: scanner.py'nin buldugu (ya da senin verdigin) coinler icin
COIN BASINA TEK model istegiyle Turkce rapor uretir.
Kota dolarsa otomatik olarak siradaki saglayiciya gecer
(Gemini -> Mistral -> Groq -> GitHub Models -> OpenRouter; sadece .env'de anahtari olanlar).

Kullanim:
    python analyze.py                      # once tarar, bulunan coinleri analiz eder
    python analyze.py --symbols S          # istedigin coini analiz et (S, APT, SOL ...)
    python analyze.py --symbols S,APT --interval 1h

Veriler: Binance (fiyat), Google News / Yahoo (haber basliklari), CoinGecko (coin adi).
Bu bir yatirim tavsiyesi degildir.
"""
import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import quote

import pandas as pd
import requests

from scanner import detect, klines, top_symbols

# ------------------------------------------------------------------ .env
def load_env(path=".env"):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8", errors="ignore"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and v and k not in os.environ:
            os.environ[k] = v


# ------------------------------------------------------------------ veri
def rsi14(close):
    d = close.diff()
    up = d.clip(lower=0).rolling(14).mean()
    dn = (-d.clip(upper=0)).rolling(14).mean()
    rs = up / dn
    return float((100 - 100 / (1 + rs)).iloc[-1])


def coin_name(ticker):
    """CoinGecko'dan ticker'a karsilik gelen en buyuk coinin adini bul (anahtarsiz)."""
    try:
        r = requests.get("https://api.coingecko.com/api/v3/search",
                         params={"query": ticker}, timeout=15)
        coins = [c for c in r.json().get("coins", []) if c.get("symbol", "").upper() == ticker.upper()]
        coins.sort(key=lambda c: c.get("market_cap_rank") or 10**9)
        if coins:
            return coins[0]["name"]
    except Exception:
        pass
    return ticker


def parse_rss(text, limit):
    out = []
    root = ET.fromstring(text)
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        date = (it.findtext("pubDate") or "").strip()[:16]
        if title:
            out.append(f"{date} | {title}")
        if len(out) >= limit:
            break
    return out


def news_headlines(name, ticker, limit=8):
    q = quote(f'"{name}" crypto' if name != ticker else f"{ticker} crypto coin")
    urls = [f"https://news.google.com/rss/search?q={q}+when:7d&hl=en-US&gl=US&ceid=US:en",
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}-USD&region=US&lang=en-US"]
    for u in urls:
        try:
            r = requests.get(u, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 200:
                items = parse_rss(r.text, limit)
                if items:
                    return items
        except Exception:
            continue
    return []


def build_prompt(sym, name, df, interval, sig, headlines):
    close = df["close"]
    ma7, ma25, ma99 = (close.rolling(n).mean().iloc[-1] for n in (7, 25, 99))
    band_lo, band_hi = min(ma7, ma25, ma99), max(ma7, ma25, ma99)
    spread = (band_hi - band_lo) / close.iloc[-1] * 100
    last = float(close.iloc[-1])
    win = df.tail(48)
    chg = (last / float(win["close"].iloc[0]) - 1) * 100
    if sig:
        bt = df["open_time"].iloc[sig["break_index"]].strftime("%d.%m %H:%M UTC")
        sig_txt = (f"VAR: yukselis sonrasi MA7, MA25'i asagi kesti, fiyat MA bolgesinin altina kirdi "
                   f"(kirilim {bt}, onceki yukselis %{sig['rise_pct']}, MA sikismasi %{sig['spread_pct']})")
    else:
        sig_txt = "YOK (kural su an tam olarak saglanmiyor)"
    news = "\n".join("- " + h for h in headlines) if headlines else "(haber basligi bulunamadi)"
    return f"""Coin: {name} ({sym}), zaman dilimi: {interval}
Son fiyat: {last:.6g} USDT
Son 48 mumda degisim: %{chg:+.1f} | en yuksek {float(win['high'].max()):.6g} | en dusuk {float(win['low'].min()):.6g}
MA7={ma7:.6g}  MA25={ma25:.6g}  MA99={ma99:.6g}  | MA sikismasi: %{spread:.2f}
Fiyatin MA bolgesine gore konumu: {"bolgenin ALTINDA" if last < band_lo else "bolgenin USTUNDE" if last > band_hi else "bolgenin ICINDE"}
RSI(14): {rsi14(close):.0f}
Tarayici dusus sinyali: {sig_txt}

Son haber basliklari:
{news}
"""


SYSTEM = """Sen temkinli bir kripto piyasa analistisin. Sadece sana verilen verileri kullan.
Verilmeyen haber, rakam ya da olay UYDURMA. Haber basliklari coinle ilgisiz gorunuyorsa
veya yoksa bunu acikca soyle. Turkce yaz, kisa ve net ol. Kesin tahmin verme, olasiliklardan bahset.
Cikti formati (basliklar aynen):
1) Teknik durum (2-3 cumle)
2) Haber ve duygu (2-3 cumle; basliklara gore)
3) Dusus uyarisi destekleniyor mu? (Evet / Hayir / Belirsiz) ve nedeni (2 cumle)
4) Riskler ve dikkat edilecekler (en fazla 3 madde)
Son satir: "Not: Yatirim tavsiyesi degildir."
"""

# ------------------------------------------------------------------ saglayicilar
_gem_model = {}


def pick_gemini_model(key):
    if "m" in _gem_model:
        return _gem_model["m"]
    model = "gemini-flash-lite-latest"
    try:
        r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                         params={"pageSize": 200}, headers={"x-goog-api-key": key}, timeout=20)
        names = []
        for m in r.json().get("models", []):
            n = m["name"].split("/")[-1]
            if ("flash-lite" in n and "generateContent" in m.get("supportedGenerationMethods", [])
                    and not any(x in n for x in ("preview", "exp", "tts", "image", "live", "audio"))):
                names.append(n)
        if names:
            model = sorted(names, reverse=True)[0]
    except Exception:
        pass
    _gem_model["m"] = model
    return model


def call_gemini(prompt, key):
    model = pick_gemini_model(key)
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": key},
        json={"systemInstruction": {"parts": [{"text": SYSTEM}]},
              "contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1200}},
        timeout=90)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    parts = r.json()["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts), f"Gemini ({model})"


def openai_compat(url, key, model, label):
    def _call(prompt, _key=key):
        r = requests.post(url, headers={"Authorization": f"Bearer {_key}"},
                          json={"model": model, "temperature": 0.3, "max_tokens": 1200,
                                "messages": [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": prompt}]},
                          timeout=90)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        return r.json()["choices"][0]["message"]["content"], f"{label} ({model})"
    return _call


def providers():
    p = []
    k = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if k:
        p.append(("Gemini", lambda prompt, k=k: call_gemini(prompt, k)))
    k = os.environ.get("MISTRAL_API_KEY")
    if k:
        p.append(("Mistral", openai_compat("https://api.mistral.ai/v1/chat/completions", k,
                                           os.environ.get("MISTRAL_MODEL", "mistral-small-latest"), "Mistral")))
    k = os.environ.get("GROQ_API_KEY")
    if k:
        p.append(("Groq", openai_compat("https://api.groq.com/openai/v1/chat/completions", k,
                                        os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"), "Groq")))
    k = os.environ.get("GITHUB_MODELS_TOKEN")
    if k:
        p.append(("GitHub Models", openai_compat("https://models.github.ai/inference/chat/completions", k,
                                                 os.environ.get("GITHUB_MODEL", "openai/gpt-4o-mini"),
                                                 "GitHub Models")))
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        p.append(("OpenRouter", openai_compat("https://openrouter.ai/api/v1/chat/completions", k,
                                              os.environ.get("OPENROUTER_MODEL",
                                                             "meta-llama/llama-3.3-70b-instruct:free"),
                                              "OpenRouter")))
    return p


def ask_llm(prompt, provs, rounds=2, wait=45):
    """Sirayla dener; hepsi basarisiz olursa bekleyip bir tur daha dener."""
    errors = []
    for rd in range(rounds):
        for name, fn in provs:
            try:
                return fn(prompt)
            except Exception as e:
                errors.append(f"{name}: {e}")
                print(f"   ! {name} basarisiz ({e}), siradakine geciliyor...")
        if rd < rounds - 1:
            print(f"   Hepsi dolu gorunuyor, {wait} sn bekleniyor...")
            time.sleep(wait)
    raise RuntimeError("Hicbir saglayici yanit vermedi: " + "; ".join(errors[-len(provs):]))


# ------------------------------------------------------------------ ana akis
def analyze_symbol(sym, interval, provs, sig=None, df=None):
    ticker = sym[:-4]
    if df is None:
        df = klines(sym, interval, limit=300)
    if sig is None:
        sig = detect(df)
    name = coin_name(ticker)
    heads = news_headlines(name, ticker)
    prompt = build_prompt(sym, name, df, interval, sig, heads)
    text, used = ask_llm(prompt, provs)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report = (f"# {name} ({sym}) - {interval}\nTarih: {stamp} | Model: {used} | "
              f"Haber basligi sayisi: {len(heads)}\n\n{text.strip()}\n")
    os.makedirs("reports", exist_ok=True)
    path = os.path.join("reports", f"{sym}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.md")
    open(path, "w", encoding="utf-8").write(report)
    return report, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", help="virgullu coin listesi, orn: S,APT,SOL")
    p.add_argument("--interval", default="30m")
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--max-spread", type=float, default=3.0)
    a = p.parse_args()

    load_env()
    provs = providers()
    if not provs:
        print("HATA: .env dosyasinda hic API anahtari bulunamadi (GOOGLE_API_KEY vb.).")
        sys.exit(1)
    print("Kullanilacak saglayicilar (sirayla):", " -> ".join(n for n, _ in provs))

    jobs = []
    if a.symbols:
        for s in a.symbols.split(","):
            s = s.strip().upper()
            jobs.append((s if s.endswith("USDT") else s + "USDT", None, None))
    else:
        syms, base = top_symbols(a.top)
        print(f"Taraniyor ({len(syms)} coin, {a.interval})...")
        for s, _, _ in syms:
            try:
                df = klines(s, a.interval, limit=300)
                sig = detect(df, max_spread_pct=a.max_spread)
            except Exception:
                continue
            if sig:
                jobs.append((s, df, sig))
            time.sleep(0.1)
        if not jobs:
            print("Su an kurala uyan coin yok. Elle coin vermek icin: python analyze.py --symbols S")
            return
        print("Bulunan coinler:", ", ".join(j[0] for j in jobs))

    for sym, df, sig in jobs:
        print(f"\n>>> {sym} analiz ediliyor...")
        try:
            report, path = analyze_symbol(sym, a.interval, provs, sig, df)
        except Exception as e:
            print(f"   {sym} atlandi: {e}")
            continue
        print("\n" + report)
        print(f"(Rapor kaydedildi: {path})")


if __name__ == "__main__":
    main()