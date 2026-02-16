import pandas as pd
import numpy as np
from collections import defaultdict

# ── Data loading & preparation ──────────────────────────────────────────────
df = pd.read_csv('/home/nckh2/qa/stock_forecasts_no_patch_full_ohlcv.csv', index_col=0, parse_dates=True)

coins = sorted(set(col.split('_')[0] for col in df.columns if '_' in col))
close_true  = {c: f'{c}_close_true'     for c in coins}
close_pred  = {c: f'{c}_close_pred' for c in coins}
open_true   = {c: f'{c}_open_true'      for c in coins}

df = df.copy()
for c in coins:
    df[f'{c}_close_prev'] = df[close_true[c]].shift(1)
df = df.iloc[1:].copy()   # drop row with no previous close

# ── Config ───────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 10000.0
MIN_PRED_RET    = 0.0035
MAX_EXPOSURE    = 1.0
TRANS_COST_BPS  = 5.0
N_RANK          = 3

# ── Return helpers ───────────────────────────────────────────────────────────
def ret_full(row, c):
    prev = row[f'{c}_close_prev']
    cl   = row[close_true[c]]
    return (cl - prev) / prev if prev > 0 else 0

def ret_intraday(row, c):
    opn = row[open_true[c]]
    cl  = row[close_true[c]]
    return (cl - opn) / opn if opn > 0 else 0

# ── Strategy signal functions ───────────────────────────────────────────────
def sig_long_only_all(row):
    sigs = []
    for c in coins:
        pr = (row[close_pred[c]] - row[open_true[c]]) / row[open_true[c]] if row[open_true[c]] > 0 else 0
        if pr > MIN_PRED_RET:
            sigs.append((c, 1.0, True))
    if not sigs: return []
    w = 1.0 / len(sigs)
    return [(c, w, True) for c, _, _ in sigs]

def sig_top_n_mom(row):
    pred = {}
    for c in coins:
        pr = (row[close_pred[c]] - row[open_true[c]]) / row[open_true[c]] if row[open_true[c]] > 0 else 0
        pred[c] = pr
    if not pred: return []
    top = sorted(pred.items(), key=lambda x: x[1], reverse=True)[:N_RANK]
    valid = [(c, 1.0, True) for c, r in top if r > MIN_PRED_RET]
    if not valid: return []
    w = 1.0 / len(valid)
    return [(c, w, True) for c, _, _ in valid]

def sig_long_short(row):
    pred = {}
    for c in coins:
        pr = (row[close_pred[c]] - row[f'{c}_close_prev']) / row[f'{c}_close_prev'] if row[f'{c}_close_prev'] > 0 else 0
        pred[c] = pr
    if not pred: return []
    srt = sorted(pred.items(), key=lambda x: x[1], reverse=True)
    longs  = [(c, 1.0, True)  for c, r in srt[:N_RANK]   if r > MIN_PRED_RET]
    shorts = [(c, -1.0, False) for c, r in srt[-N_RANK:] if r < -MIN_PRED_RET]
    all_sig = longs + shorts
    if not all_sig: return []
    gross = sum(abs(w) for _, w, _ in all_sig)
    norm = 1.0 / gross
    return [(c, w * norm, lng) for c, w, lng in all_sig]

# ── Simulation ───────────────────────────────────────────────────────────────
def run_strategy(sig_func, name, ret_func):
    capital = INITIAL_CAPITAL
    peak = capital
    max_dd = 0.0
    daily_rets = []
    trade_days = 0

    coin_track = defaultdict(lambda: {'pnl': 0.0, 'days': 0, 'wins': 0, 'gross_rets': []})

    for _, row in df.iterrows():
        signals = sig_func(row)
        if not signals:
            dr = 0.0
        else:
            trade_days += 1
            tot_abs = sum(abs(w) for _, w, _ in signals)
            scale = min(tot_abs, MAX_EXPOSURE) / tot_abs if tot_abs > 0 else 0
            dr = 0.0
            for c, w, is_long in signals:
                gr = ret_func(row, c)
                signed = gr if is_long else -gr
                tcost = (TRANS_COST_BPS / 10000) * abs(w)
                net = signed - tcost
                contrib = net * abs(w) * scale
                dr += contrib

                # per coin
                ct = coin_track[c]
                ct['pnl'] += contrib * capital
                ct['days'] += 1
                ct['gross_rets'].append(signed)
                if signed > 0:
                    ct['wins'] += 1

        capital *= (1 + dr)
        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
        daily_rets.append(dr)

    # Portfolio stats
    tot_ret = (capital / INITIAL_CAPITAL) - 1
    sharpe = (np.mean(daily_rets) / np.std(daily_rets)) * np.sqrt(365) if np.std(daily_rets) > 0 else 0
    calmar = tot_ret / max_dd if max_dd > 0 else 0

    print(f"\n{name}")
    print(f"  Total Return : {tot_ret:8.2%}")
    print(f"  Sharpe       : {sharpe:8.2f}")
    print(f"  Max DD       : {max_dd:8.2%}")
    print(f"  Calmar       : {calmar:8.2f}")
    print(f"  Final $      : {capital:9,.0f}")
    print(f"  Trade days   : {trade_days:4d} / {len(df)}")

    # Per-coin table
    rows = []
    for c, st in coin_track.items():
        if st['days'] == 0: continue
        hit = st['wins'] / st['days'] if st['days'] > 0 else 0
        avg_gr = np.mean(st['gross_rets']) if st['gross_rets'] else 0
        cum_gr = np.prod(1 + np.array(st['gross_rets'])) - 1
        rows.append({
            'Coin': c,
            'PnL $': round(st['pnl'], 1),
            'Days': st['days'],
            'Hit': f"{hit:4.1%}",
            'Avg ret/day': f"{avg_gr:6.2%}",
            'Cum ret': f"{cum_gr:6.2%}",
        })

    if rows:
        pdf = pd.DataFrame(rows).sort_values('PnL $', ascending=False).set_index('Coin')
        print(pdf.round(2))
    else:
        print("  No positions taken")

    return {
        'name': name,
        'Total Return': tot_ret,
        'Sharpe': sharpe,
        'Max DD': max_dd,
        'Calmar': calmar,
        'Final $': capital
    }

# ── Run all ──────────────────────────────────────────────────────────────────
strategies = [
    (sig_long_only_all,     "1. Intraday Long-Only (all > thresh)", ret_intraday),
    (sig_top_n_mom,         "2. Intraday Top-N Momentum",           ret_intraday),
    (sig_long_short,        "3. Long-Short (top/bottom N)",         ret_full),
]

summary_rows = []
for sig_f, name, ret_f in strategies:
    res = run_strategy(sig_f, name, ret_f)
    summary_rows.append(res)

# Final quick comparison
print("\n" + "-"*70)
print("SUMMARY COMPARISON")
sum_df = pd.DataFrame(summary_rows).set_index('name')
sum_df = sum_df[['Total Return', 'Sharpe', 'Max DD', 'Calmar', 'Final $']]
sum_df['Total Return'] = sum_df['Total Return'].apply(lambda x: f"{x:7.2%}")
sum_df['Max DD']       = sum_df['Max DD'].apply(lambda x: f"{x:7.2%}")
print(sum_df.round(2))
print("-"*70)