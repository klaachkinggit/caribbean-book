# Caribbean Poker Simulation

This project simulates and analyzes Caribbean Poker using a Jupyter notebook and a small Python helper module for poker hand evaluation.

The notebook is written mostly in French and covers game mechanics, hand comparison, expected value analysis, and Monte Carlo simulation.

## Project Structure

```text
carribean.ipynb       # Main notebook with the simulation and analysis
helpers/
  poker.py            # Poker hand ranking and comparison helpers
```

## Requirements

- Python 3
- Jupyter Notebook
- matplotlib
- numpy
- scipy

Install the Python dependencies with:

```bash
pip install jupyter matplotlib numpy scipy
```

## Run

Start Jupyter Notebook:

```bash
jupyter notebook carribean.ipynb
```

Then run the notebook cells sequentially.

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
