"""TopstepX 50K combine simulator. Do not modify.

strategy.signals(df) -> int array, target MNQ contracts in [-50, 50] per row, decided on that row's close.
Fills at next row's open +/- slippage. Rules: $3,000 target, $1,000 daily loss (flatten, day over),
$2,000 trailing max loss on EOD high (locks at $50,000), flat by 15:10 CT, consistency
(best day < 50% of profit), min 2 days. Score = min(pass_rate first half, pass_rate second half)
of rolling 30-session attempts, plus a small speed bonus.
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from numba import njit

# ponytail: MNQ priced off NQ 1s data (same index, tighter-tracking); fee is an estimate, confirm vs TopstepX
PT_VALUE, SLIP_PTS, COMM_SIDE = 2.0, 0.25, 0.50    # MNQ $/pt, 1 tick/side, $/contract/side
START, TARGET, DLL, MLL, MAX_CT = 50_000.0, 3_000.0, 1_000.0, 2_000.0, 50
FLAT_SEC, WINDOW = 15 * 3600 + 10 * 60, 30          # 15:10 CT; attempt length in sessions


@njit(cache=True)
def run_days(sess, ct_sec, o, h, l, c, target, n_sess):
    day_pnl, day_min, trades = np.zeros(n_sess), np.zeros(n_sess), np.zeros(n_sess, np.int64)
    pos, cash, entry_mtm, locked = 0, 0.0, 0.0, False
    for i in range(len(c)):
        s = sess[i]
        if i == 0 or s != sess[i - 1]:
            pos, cash, locked = 0, 0.0, False
        last = i == len(c) - 1 or sess[i + 1] != s
        want = 0 if locked or (ct_sec[i] >= FLAT_SEC and ct_sec[i] < 17 * 3600) else target[i - 1] if i > 0 and sess[i - 1] == s else 0
        want = max(-MAX_CT, min(MAX_CT, want))
        if want != pos:
            d = want - pos
            px = o[i] + SLIP_PTS * (1 if d > 0 else -1)
            cash -= d * px * PT_VALUE + abs(d) * COMM_SIDE
            pos = want
            trades[s] += abs(d)
        worst = cash + pos * (l[i] if pos > 0 else h[i]) * PT_VALUE
        day_min[s] = min(day_min[s], worst)
        if worst <= -DLL and not locked:          # DLL hit: flatten at the limit, day over
            cash = -DLL - abs(pos) * (SLIP_PTS * PT_VALUE + COMM_SIDE)
            trades[s] += abs(pos)
            pos, locked = 0, True
        if last and pos != 0:                     # session end: flatten at close
            cash += pos * (c[i] - SLIP_PTS * np.sign(pos)) * PT_VALUE - abs(pos) * COMM_SIDE
            trades[s] += abs(pos)
            pos = 0
        day_pnl[s] = cash + pos * c[i] * PT_VALUE
    return day_pnl, day_min, trades


def attempts(day_pnl, day_min, lo, hi):
    """Rolling combine attempts starting at each session in [lo, hi - WINDOW]. Returns (passed, days)."""
    res = []
    for st in range(lo, hi - WINDOW + 1):
        bal, peak_eod, best, days, out = START, START, 0.0, 0, (False, WINDOW)
        for s in range(st, st + WINDOW):
            thr = min(peak_eod - MLL, START)
            if bal + day_min[s] <= thr:
                break
            bal += day_pnl[s]
            days += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - START
            if profit >= TARGET and days >= 2 and best < 0.5 * profit:
                out = (True, days)
                break
        res.append(out)
    return res


def score(day_pnl, day_min):
    n, mid = len(day_pnl), len(day_pnl) // 2
    halves = [attempts(day_pnl, day_min, 0, mid), attempts(day_pnl, day_min, mid, n)]
    rates = [np.mean([p for p, _ in a]) if a else 0.0 for a in halves]
    passed = [d for a in halves for p, d in a if p]
    speed = 1 - np.mean(passed) / WINDOW if passed else 0.0
    return min(rates) + 0.01 * speed, rates, passed


def lookahead_check(signals, df, rng, n=3):
    """Signals on a prefix must equal signals on full data for that prefix."""
    full = np.asarray(signals(df))
    for cut in rng.integers(len(df) // 10, len(df), n):
        part = np.asarray(signals(df.iloc[:cut].copy()))
        if not np.array_equal(part[:cut - 1], full[:cut - 1]):
            bad = int(np.argmax(part[:cut - 1] != full[:cut - 1]))
            sys.exit(f"[eval] LOOKAHEAD: signal at row {bad} changes when future data after row {cut} is removed")
    return full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser(os.environ.get("NQ_DATA", "~/hive-data/nq/train.parquet")))
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from strategy import signals

    df = pd.read_parquet(a.data)
    sess = pd.factorize(df.session, sort=True)[0]
    check = df[df.session.isin(df.session.unique()[:15])]
    lookahead_check(signals, check, np.random.default_rng(0))
    target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
    assert len(target) == len(df), "signals() must return one value per row"
    day_pnl, day_min, trades = run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, sess.max() + 1)
    sc, rates, passed = score(day_pnl, day_min)
    print("---")
    print(f"score:          {sc:.4f}")
    print(f"pass_rate_h1:   {rates[0]:.4f}")
    print(f"pass_rate_h2:   {rates[1]:.4f}")
    print(f"avg_days_pass:  {np.mean(passed) if passed else float('nan'):.1f}")
    print(f"sessions:       {len(day_pnl)}")
    print(f"total_pnl:      {day_pnl.sum():.0f}")
    print(f"win_days:       {(day_pnl > 0).mean():.3f}")
    print(f"worst_day:      {day_pnl.min():.0f}")
    print(f"contracts_traded: {trades.sum()}")


if __name__ == "__main__":
    main()
