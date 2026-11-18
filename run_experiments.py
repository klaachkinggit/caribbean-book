"""Re-run the Caribbean Stud experiments with the corrected engine.

  python run_experiments.py            # writes results/results.json and prints a summary

Experiments
  1. The notebook configuration: 7 players x 500 rounds, seed 100, Monte Carlo over
     10 seeds, players who only see their own hand (a real casino).
  2. House edge of the EV-optimal policy, estimated over many rounds. Two estimators
     on the same deals: realized P&L (high variance) and the exact EV of the chosen
     action (same mean, much lower variance). Both clustered by round.
  3. Value of information: the same deals when every player sees the whole table.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from helpers import caribbean as cb

PLAYERS = 7
BIG_ROUNDS = int(os.environ.get("BIG_ROUNDS", 20000))
CHUNK = 500


def _run(args):
    info, seed, rounds = args
    return cb.simulate_table(PLAYERS, rounds, info, seed)


def _many(info, seeds, rounds, workers):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_run, [(info, s, rounds) for s in seeds]))


def main():
    cb.build_tables()                       # build the cache once before forking
    workers = os.cpu_count() or 4
    out = {}
    t0 = time.time()

    # 1. notebook configuration
    nb = cb.simulate_table(PLAYERS, 500, "own", seed=100)
    final = 1000 + nb["pnl"].sum(axis=0)
    mc = _many("own", range(100, 110), 500, workers)
    mc_profit = np.array([r["pnl"].sum(axis=0) for r in mc])          # (10 seeds, 7 players)
    out["notebook_config"] = {
        "final_capital_seed100": final.tolist(),
        "call_rate": float(nb["called"].mean()),
        "mc_mean_profit_per_player_500_rounds": float(mc_profit.mean()),
        "mc_share_of_players_in_profit": float((mc_profit > 0).mean()),
        "mc_profit_min_max": [float(mc_profit.min()), float(mc_profit.max())],
    }

    # 2 + 3. large paired run: same seeds -> same deals under both information models
    seeds = list(range(1000, 1000 + BIG_ROUNDS // CHUNK))
    own = _many("own", seeds, CHUNK, workers)
    table = _many("table", seeds, CHUNK, workers)
    for name, runs in (("own", own), ("table", table)):
        pnl = np.vstack([r["pnl"] for r in runs])
        ev = np.vstack([r["ev"] for r in runs])
        called = np.vstack([r["called"] for r in runs])
        out[name] = {"realized": cb.summarize(pnl), "exact_ev": cb.summarize(ev),
                     "call_rate": float(called.mean())}
    d_ev = np.vstack([r["ev"] for r in table]) - np.vstack([r["ev"] for r in own])
    out["value_of_table_information"] = cb.summarize(d_ev)
    out["runtime_s"] = round(time.time() - t0, 1)
    out["big_rounds"] = BIG_ROUNDS

    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    json.dump(out, sys.stdout, indent=2, default=float)


if __name__ == "__main__":
    main()
