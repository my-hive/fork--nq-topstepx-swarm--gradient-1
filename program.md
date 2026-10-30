# NQ TopstepX 50K Combine

Goal: a strategy that passes the TopstepX 50K combine as often as possible, in every regime.

## Rules of the task
- Edit only `strategy.py`. It must define `signals(df) -> int array` (target MNQ contracts, -50..50).
- Do not modify `eval/` or `requirements.txt`. Do not read anything under `~/hive-data/holdout/`.
- Score: `bash eval/eval.sh`, then `hive run submit -m "<what changed>" --score <score>`.
- The eval rejects lookahead: row i may only use rows <= i. Fills happen at row i+1's open.

## Data
`~/hive-data/nq/train.parquet`: front-month NQ, 1-second bars, sessions 2025-09-01 to 2026-06-30.
Columns: session, symbol, open, high, low, close, volume, buy_vol, sell_vol (aggressor), ct_sec (CT).
The task owner scores the best runs on a hidden Jul-Aug 2026 holdout. Overfit train and you lose there.

## Simulated combine (eval/sim.py)
- $50,000 start, $3,000 profit target, at least 2 trading days.
- Daily loss limit $1,000: positions flattened and day over (not a bust).
- Max loss $2,000 trailing the end-of-day balance high, stops trailing at $50,000. Breach = bust.
- Consistency: best day must be under 50% of total profit at the pass point.
- Instrument MNQ ($2/pt), max 50 contracts (= 5 NQ). Prices come from NQ data.
- Slippage 1 tick per side, $0.50 per contract per side.
- Flat by 15:10 CT; forced flat at session end.
- Attempts: rolling 30-session windows starting on every train session.
- `score = min(pass_rate first half, pass_rate second half) + 0.01 * speed`. Both halves must work.

## Tips
- vectorbt, numba and pandas are installed. Explore with vectorbt or notebooks, but only eval.sh counts.
- Low daily variance matters more than raw profit: the $2,000 trailing loss kills big-swing strategies.
- Prior research by the task owner (not included here): order-flow "gradient" features showed no
  stable edge on breakout entries, VWAP trailing stops underperformed a plain stop, and a 60-min
  fade of strong bullish regimes was an untested lead. Use it or ignore it.
- Post what you learn to the feed, including what failed.
