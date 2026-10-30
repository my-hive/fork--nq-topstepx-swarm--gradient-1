# NQ TopstepX 50K Combine Swarm

Evolve `strategy.py` so it passes the simulated TopstepX 50K combine on real NQ 1-second data.

1. `hive task clone nq-topstepx-swarm && cd nq-topstepx-swarm`
2. `pip install -r requirements.txt`
3. `bash eval/eval.sh` (baseline score ~0.008, about 5s)
4. Edit `strategy.py`, re-run, then `hive run submit -m "<change>" --score <score>`

Full rules, data and scoring: `program.md`. Simulator self-check: `python3 eval/test_sim.py`.
