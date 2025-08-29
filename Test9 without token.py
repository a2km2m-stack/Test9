
#!/usr/bin/env python3
# telegram_crypto_bot_multi_school.py
# Python 3.8+
# Requirements: requests, pandas

import time
import requests
import json
import os
import pandas as pd
from datetime import datetime
import traceback
import math

# ------------------ CONFIG ------------------
BOT_TOKEN ""
POLL_TIMEOUT = 30
MAX_MESSAGE_CHUNK = 3500
USERS_FILE = "users.json"

TG_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
TG_GETUPDATES = TG_API + "/getUpdates"
TG_SENDMESSAGE = TG_API + "/sendMessage"

# ------------------ Load Users (username -> role) ------------------
def load_users_dict():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r") as f:
                data = json.load(f)
                # support older shape or dict
                if isinstance(data, dict) and "users" not in data:
                    # maybe dict of username->role
                    return data
                users_list = data.get("users", []) if isinstance(data, dict) else []
                return {u["username"]: u["role"] for u in users_list}
        except Exception:
            return {}
    return {}

users = load_users_dict()

def save_users():
    with open(USERS_FILE, "w") as f:
        json.dump({"users": [{"username": k, "role": v} for k, v in users.items()]}, f, indent=2)

# ------------------ Utilities ------------------
def safe_get(url, params=None, max_retries=3):
    for attempt in range(1, max_retries+1):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                return {"error": True, "status_code": r.status_code, "text": r.text}
        except Exception:
            time.sleep(1 * attempt)
    return {"error": True, "exception": "max_retries_exceeded"}

def send_text(chat_id, text):
    # chunk
    while text:
        chunk = text[:MAX_MESSAGE_CHUNK]
        if len(text) > MAX_MESSAGE_CHUNK:
            last_nl = chunk.rfind("\n")
            if last_nl > int(MAX_MESSAGE_CHUNK * 0.6):
                chunk = chunk[:last_nl]
        payload = {"chat_id": chat_id, "text": chunk}
        try:
            requests.post(TG_SENDMESSAGE, data=payload, timeout=10)
        except Exception as e:
            print("Send error:", e)
        text = text[len(chunk):]

# ------------------ Binance OHLCV ------------------
def fetch_ohlcv(symbol="BTC", interval="1d", limit=500):
    # Binance supports intervals like 1m,3m,5m,15m,1h,4h,1d,1w,1M
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper() + "USDT", "interval": interval, "limit": limit}
    res = safe_get(url, params=params)
    if not res or "error" in res:
        return None, res
    rows = []
    for k in res:
        rows.append({
            "time": datetime.utcfromtimestamp(k[0]/1000),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5])
        })
    df = pd.DataFrame(rows)
    return df, None

# ------------------ Indicators ------------------
def sma(series, window):
    return series.rolling(window=window, min_periods=1).mean()

def rsi(series, window=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/window, adjust=False).mean()
    ma_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df, window=14):
    # average true range
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=1).mean()

# ------------------ Swing points & zones ------------------
def find_swing_points(df, lookback=5):
    highs, lows = [], []
    for i in range(lookback, len(df)-lookback):
        window_high = df['high'].iloc[i-lookback:i+lookback+1]
        window_low  = df['low'].iloc[i-lookback:i+lookback+1]
        if df['high'].iloc[i] == window_high.max():
            highs.append((i, float(df['high'].iloc[i]), df['time'].iloc[i]))
        if df['low'].iloc[i] == window_low.min():
            lows.append((i, float(df['low'].iloc[i]), df['time'].iloc[i]))
    return highs, lows

def cluster_levels(levels, tol=0.01):
    # cluster price levels that are within tol (fraction) of each other
    clusters = []
    for lv in sorted(levels):
        placed=False
        for c in clusters:
            # c stores centroid
            if abs(lv - c['centroid']) / c['centroid'] <= tol:
                c['members'].append(lv)
                c['centroid'] = sum(c['members'])/len(c['members'])
                placed=True
                break
        if not placed:
            clusters.append({'centroid': lv, 'members':[lv]})
    # return centroids sorted descending for supply, ascending for demand
    return [c['centroid'] for c in clusters]

def detect_supply_demand_zones(df, highs, lows):
    # supply zones from highs clusters, demand from lows clusters
    high_levels = [h[1] for h in highs]
    low_levels = [l[1] for l in lows]
    supply = cluster_levels(high_levels, tol=0.01) if high_levels else []
    demand = cluster_levels(low_levels, tol=0.01) if low_levels else []
    # choose recent ones: take up to 3 nearest
    supply_recent = sorted(supply, reverse=True)[:3]
    demand_recent = sorted(demand)[:3]
    return supply_recent, demand_recent

# ------------------ Wyckoff simple heuristic ------------------
def wyckoff_signal(df):
    # simple heuristic:
    # accumulation: price range narrow (low std), volume increasing recently, RSI rising from oversold
    res = None
    try:
        recent = df.tail(50)
        hist = df['close']
        std_recent = recent['close'].pct_change().std()
        std_hist = hist.pct_change().std() if len(hist)>50 else std_recent
        vol_recent = recent['volume'].mean()
        vol_prev = df.tail(150).head(100)['volume'].mean() if len(df)>=150 else vol_recent
        rsi_now = rsi(df['close']).iloc[-1]
        rsi_10 = rsi(df['close']).iloc[-10:].mean()
        # accumulation if volatility low and volume rising and RSI rising from low
        if std_recent < std_hist*0.6 and vol_recent > vol_prev*1.1 and rsi_10 > (rsi_now - 2):
            res = "Possible Accumulation"
        # distribution opposite conditions
        if std_recent < std_hist*0.6 and vol_recent > vol_prev*1.1 and rsi_10 < (rsi_now + 2) and rsi_now>60:
            res = "Possible Distribution"
    except Exception:
        res = None
    return res

# ------------------ Targets calculation ------------------
def fib_extensions(low, high):
    diff = high - low
    return [high + 0.382*diff, high + 0.618*diff, high + 1.0*diff]

def choose_up_targets(fib_list, last_close, max_n=3):
    candidates = [t for t in fib_list if t > last_close]
    candidates = sorted(candidates)
    return candidates[:max_n]

def choose_down_targets(demand_zones, lows_list, last_close, df):
    # combine demand_zones and last swing lows; choose those < price, sort by proximity asc
    candidates = []
    # from demand_zones (centroids)
    for z in demand_zones:
        if z < last_close:
            candidates.append(z)
    # from last swing lows (most recent)
    if lows_list:
        last_lows_vals = [l[1] for l in lows_list[-5:]]
        for lv in last_lows_vals:
            if lv < last_close:
                candidates.append(lv)
    # remove duplicates and remove SMA50/SMA200 approx
    sma50 = float(sma(df['close'],50).iloc[-1]) if len(df)>=50 else None
    sma200 = float(sma(df['close'],200).iloc[-1]) if len(df)>=200 else None
    filtered = []
    for c in sorted(set(candidates), key=lambda x: abs(last_close - x)):
        skip = False
        if sma50 is not None and math.isclose(c, sma50, rel_tol=5e-4, abs_tol=1e-6):
            skip = True
        if sma200 is not None and math.isclose(c, sma200, rel_tol=5e-4, abs_tol=1e-6):
            skip = True
        if not skip:
            filtered.append(c)
    # return closest two (closest meaning minimal distance from price)
    filtered_sorted = sorted(filtered, key=lambda x: abs(last_close - x))
    return filtered_sorted[:2]

# ------------------ Confidence and trend ------------------
def detect_trend(df):
    try:
        s20 = sma(df['close'],20).iloc[-1]
        s50 = sma(df['close'],50).iloc[-1]
        s200 = sma(df['close'],200).iloc[-1]
        if s20 > s50 > s200:
            return "bullish"
        if s20 < s50 < s200:
            return "bearish"
    except Exception:
        pass
    return "neutral"

def compute_confidence(trend, rsi_now, wyckoff_note):
    conf = 50
    if trend=="bullish": conf += 15
    if trend=="bearish": conf -= 10
    if rsi_now < 30: conf += 8
    if rsi_now > 70: conf -= 8
    if wyckoff_note and "Accumulation" in wyckoff_note:
        conf += 6
    if wyckoff_note and "Distribution" in wyckoff_note:
        conf -= 6
    return max(5, min(95, int(conf)))

# ------------------ Make report (single timeframe) ------------------
def make_report_frame(symbol, df, interval):
    # compute indicators
    df = df.copy().reset_index(drop=True)
    df['sma20'] = sma(df['close'],20)
    df['rsi14'] = rsi(df['close'],14)
    df['atr14'] = atr(df, 14)

    last_close = float(df['close'].iloc[-1])
    rsi_now = float(df['rsi14'].iloc[-1])

    highs, lows = find_swing_points(df, lookback=5)
    last_high = highs[-1][1] if highs else float(df['high'].max())
    last_low = lows[-1][1] if lows else float(df['low'].min())

    # Fib up targets
    fibs = fib_extensions(last_low, last_high)
    up_targets = choose_up_targets(fibs, last_close, max_n=3)

    # supply/demand zones
    supply_zones, demand_zones = detect_supply_demand_zones(df, highs, lows)

    # down targets (2 only)
    down_targets = choose_down_targets(demand_zones, lows, last_close, df)

    # wyckoff note
    wyck = wyckoff_signal(df)

    # trend & confidence
    trend = detect_trend(df)
    confidence = compute_confidence(trend, rsi_now, wyck)

    # Build lines
    lines = []
    lines.append(f"📊 Advanced Analysis for {symbol.upper()}")
    lines.append(f"Date (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Timeframe: {interval}")
    lines.append(f"Last Close: {last_close:.6f} USDT")
    lines.append(f"Trend: {trend} | Confidence: {confidence}%")
    lines.append("")

    # Up Targets (up to 3)
    lines.append("🟢 Up Targets")
    if up_targets:
        for i, u in enumerate(up_targets, 1):
            # annotate if matches supply zone
            note = ""
            # if close to any supply zone mark it
            for sz in supply_zones:
                if abs(u - sz) / (sz if sz!=0 else 1) < 0.02:
                    note = " - Supply zone"
                    break
            lines.append(f"- {u:.6f} (Target {i}){note}")
    else:
        lines.append("- (no reasonable up targets above price)")

    lines.append("")

    # Down Targets (Stop Loss) — only up to 2
    lines.append("🔴 Down Targets / Stop Loss")
    if down_targets:
        for i, d in enumerate(down_targets, 1):
            # annotate if close to demand zone
            note = ""
            for dz in demand_zones:
                if abs(d - dz) / (dz if dz!=0 else 1) < 0.02:
                    note = " - Demand zone"
                    break
            lines.append(f"- {d:.6f} (Stop loss {i}){note}")
    else:
        lines.append("- (no reasonable stop losses found below price)")

    lines.append("")
    # Indicators summary (show SMA20, SMA50, SMA200 if available)
    s20 = float(sma(df['close'],20).iloc[-1])
    s50 = float(sma(df['close'],50).iloc[-1]) if len(df)>=50 else None
    s200 = float(sma(df['close'],200).iloc[-1]) if len(df)>=200 else None
    sma_line = f"SMA20={s20:.6f}"
    if s50 is not None:
        sma_line += f", SMA50={s50:.6f}"
    if s200 is not None:
        sma_line += f", SMA200={s200:.6f}"
    lines.append(f"Indicators: {sma_line}, RSI14={rsi_now:.2f}, ATR14={float(df['atr14'].iloc[-1]):.6f}")
    lines.append("")
    if wyck:
        lines.append(f"🧠 Wyckoff: {wyck}")
        lines.append("")
    lines.append("⚠️ Note: purely mechanical multi-school analysis. Not financial advice.")
    return "\n".join(lines)

# ------------------ Command handlers ------------------
def handle_add(chat_id, requester, username, role="user"):
    # only admin can add
    if users.get(requester) != "admin":
        send_text(chat_id, "❌ Only admin can add users.")
        return
    if username in users:
        send_text(chat_id, f"User @{username} already exists as {users[username]}.")
        return
    users[username] = role
    save_users()
    send_text(chat_id, f"✅ Added @{username} with role {role}.")

def handle_remove(chat_id, requester, username):
    if users.get(requester) != "admin":
        send_text(chat_id, "❌ Only admin can remove users.")
        return
    if username not in users:
        send_text(chat_id, f"User @{username} not found.")
        return
    users.pop(username)
    save_users()
    send_text(chat_id, f"✅ Removed @{username}.")

def parse_intervals_arg(arg_text):
    # accepts comma separated like "4h,1d,1w" or single like "4h"
    parts = [p.strip() for p in arg_text.split(",") if p.strip()]
    # validate simple set; default map: accept common Binance intervals
    allowed = {"1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}
    clean = []
    for p in parts:
        if p in allowed:
            clean.append(p)
    return clean

def handle_analyze_command(chat_id, requester, symbol, intervals_arg=None):
    # permission
    if requester not in users:
        send_text(chat_id, "❌ You are not authorized to use this bot. Contact admin.")
        return
    # Parse intervals: default daily if not provided
    if not intervals_arg:
        intervals = ["1d"]
    else:
        # allow comma separated or single token
        if "," in intervals_arg:
            intervals = parse_intervals_arg(intervals_arg)
            if not intervals:
                intervals = ["1d"]
        else:
            # single token like "4h" or "1d"
            if intervals_arg in {"1m","3m","5m","15m","30m","1h","2h","4h","6h","8h","12h","1d","3d","1w","1M"}:
                intervals = [intervals_arg]
            else:
                # maybe user passed symbol as second arg and no timeframe; default daily
                intervals = ["1d"]
    # Build multi-frame report (concatenate)
    full_report_parts = []
    for tf in intervals:
        df, err = fetch_ohlcv(symbol, tf, limit=500)
        if err or df is None or len(df) < 30:
            send_text(chat_id, f"❌ Error fetching {symbol} {tf}: {err}")
            return
        part = make_report_frame(symbol, df, tf)
        full_report_parts.append(part)
    report_text = "\n\n".join(full_report_parts)
    send_text(chat_id, report_text)

def handle_help(chat_id):
    txt = (
        "📘 Commands:\n"
        "/analyze <SYMBOL> [INTERVAL]\n"
        "   - INTERVAL optional (default 1d). Examples: 4h, 1d, 1w or comma-list: 4h,1d\n"
        "/add <username> <role>    (admin only)  # role: admin or user\n"
        "/remove <username>        (admin only)\n"
        "/help\n"
        "\nNotes:\n- Users identified by Telegram username (without @) as stored in users.json.\n- Analysis is mechanical: SMA/RSI/Fib/Supply&Demand/Wyckoff heuristic.\n- Not financial advice."
    )
    send_text(chat_id, txt)

# ------------------ Main long-poll loop ------------------
def run_longpoll():
    offset = None
    print("Bot started (multi-school analysis).")
    while True:
        try:
            params = {"timeout": POLL_TIMEOUT}
            if offset:
                params["offset"] = offset
            resp = requests.get(TG_GETUPDATES, params=params, timeout=POLL_TIMEOUT+10)
            if resp.status_code != 200:
                time.sleep(1)
                continue
            data = resp.json()
            if not data.get("ok"):
                time.sleep(1)
                continue
            updates = data.get("result", [])
            if not updates:
                time.sleep(0.5)
                continue
            for upd in updates:
                offset = upd["update_id"] + 1
                msg = upd.get("message") or {}
                chat = msg.get("chat", {}) or {}
                chat_id = chat.get("id")
                text = (msg.get("text") or "").strip()
                from_user = msg.get("from") or {}
                username = from_user.get("username", "")
                if not text or not chat_id:
                    continue
                parts = text.split(maxsplit=2)
                cmd = parts[0].lower()
                # /analyze <SYMBOL> [INTERVAL or comma list]
                if cmd == "/analyze":
                    if len(parts) >= 2:
                        symbol = parts[1].strip().upper()
                        intervals_arg = None
                        if len(parts) == 3:
                            intervals_arg = parts[2].strip()
                        handle_analyze_command(chat_id, username, symbol, intervals_arg)
                    else:
                        send_text(chat_id, "Usage: /analyze <SYMBOL> [INTERVAL]\nExample: /analyze BTC 1d\nOr: /analyze BTC 4h,1d")
                elif cmd == "/add":
                    # /add <username> <role>
                    if len(parts) >= 3:
                        # parts[1] is username, remainder possibly role
                        sub = parts[1].strip()
                        role = parts[2].strip()
                        handle_add(chat_id, username, sub, role)
                    else:
                        send_text(chat_id, "Usage: /add <username> <role>")
                elif cmd == "/remove":
                    # /remove <username>
                    if len(parts) >= 2:
                        target = parts[1].strip()
                        handle_remove(chat_id, username, target)
                    else:
                        send_text(chat_id, "Usage: /remove <username>")
                elif cmd == "/help":
                    handle_help(chat_id)
                else:
                    # ignore unknown
                    pass
        except Exception:
            traceback.print_exc()
            time.sleep(2)

# ------------------ Run ------------------
if __name__ == "__main__":
    # ensure users file exists
    if not os.path.exists(USERS_FILE):
        # create empty file to avoid KeyErrors
        with open(USERS_FILE, "w") as f:
            json.dump({"users": []}, f, indent=2)
    # reload users variable
    users = load_users_dict()
    run_longpoll()