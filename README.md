# Caribbean Poker Simulation

This project simulates and analyzes Caribbean Poker using a Jupyter notebook and a small Python helper module for poker hand evaluation.

The notebook is written mostly in French and covers game mechanics, hand comparison, expected value analysis, and Monte Carlo simulation.

## Project Structure

```text
carribean.ipynb       # Main notebook with the simulation and analysis
helpers/
  poker.py            # Poker hand ranking and comparison helpers
  caribbean.py        # Exact call/fold EV (vectorised) and the table simulator
tests/                # Payout, scoring and EV checks against a brute-force reference
run_experiments.py    # Re-runs every experiment, writes results/results.json
```

## Requirements

- Python 3
- Jupyter Notebook
- matplotlib
- numpy
- scipy

Install the Python dependencies with:

```bash
pip install jupyter matplotlib numpy scipy pytest
```

## Run

Start Jupyter Notebook:

```bash
jupyter notebook carribean.ipynb
```

Then run the notebook cells sequentially. To reproduce the numbers below and run the tests:

```bash
python -m pytest tests -q
python run_experiments.py      # ~6 min on 8 cores
```

## What The Notebook Does

- Creates and deals poker hands
- Evaluates poker hand ranks
- Checks whether the dealer qualifies
- Compares player and dealer hands
- Calculates expected value for call decisions
- Runs configurable multi-player simulations
- Runs Monte Carlo simulations and reports confidence intervals

## Main Parameters

The notebook includes these default simulation parameters:

| Parameter | Default | Meaning |
|---|---:|---|
| `NUM_PLAYERS` | 7 | Players per game |
| `INITIAL_CAPITAL` | 1000 | Starting bankroll |
| `ANTE_BET` | 1 | Ante bet amount |
| `CALL_BET` | 2 | Call bet amount |
| `NUM_ROUNDS` | 500 | Rounds per simulation |
| `SEED` | 100 | Random seed |
| `MC_NUM_SIMULATIONS` | 10 | Monte Carlo runs |
| `MC_CONFIDENCE_LEVEL` | 0.95 | Confidence interval level |

## Helper Module

`helpers/poker.py` implements standard poker hand ranking from High Card through Royal Flush.

Important functions:

- `hand_rank(hand)`
- `compare_hands(hand1, hand2)`

These helpers are independent of the Caribbean Poker rules and are used by the notebook to compare player and dealer hands.

## Results

The EV-optimal player calls when EV(call) > -ante, computed exactly over every dealer
completion consistent with what the player can see (C(46,4) = 163,185 hands at a real
table). Over 20,000 seven-player rounds (140,000 hands), with 95% confidence intervals
clustered by round because every player faces the same dealer hand:

| Information | House edge (% of ante) | Call rate |
|---|---:|---:|
| Own hand + upcard (real casino) | **5.6%** [4.8, 6.4] | 52.1% |
| Published optimal strategy (reference) | 5.224% | ~52.2% |
| Whole table visible (7 players share cards) | **-2.3%** [-3.1, -1.5] (player edge) | 62.9% |

The house-edge estimate averages the exact EV of each chosen action, which has the same mean
as realized P&L at about half the standard error (realized P&L gives 4.8% [3.2, 6.3]).
Seeing the other 30 cards is worth about 7.9 cents per hand and turns the game in the
player's favour.

At the notebook's scale (7 players x 500 rounds, 10 seeds) results are dominated by
variance: the average player loses $34 over 500 rounds, yet 24% of player-runs end in profit.

## Corrections (November 2026)

The first version overstated the player's results. Fixed and covered by tests:

1. **Payouts in the EV formula.** A non-qualifying dealer paid +2 and a win paid 2 + 2 x
   multiplier; the rules give +1 (ante 1:1, call returned) and 1 + 2 x multiplier. The EV
   was overstated by one ante in those branches, so the player called about 80% of hands.
2. **One dealer hand per round.** Each player drew their own random dealer hand; now one
   shuffled deck produces a single dealer hand that settles every player at the table.
3. **Information model.** Every player saw all other players' cards. The default is now a
   real casino (own hand + upcard), with full-table sharing kept as an explicit option.
