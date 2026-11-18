"""Caribbean Stud: exact call/fold EV and a table simulator.

Rules (standard paytable):
  * The player antes A, sees 5 cards and the dealer's upcard, then folds (loses A)
    or calls with C = 2A.
  * The dealer qualifies with Ace-King high or better.
  * Dealer does not qualify -> ante pays 1:1, call is returned       net = +A
  * Dealer qualifies, player wins -> ante 1:1, call pays C * mult     net = +A + C*mult
  * Tie -> both bets pushed                                           net = 0
  * Dealer qualifies, player loses                                    net = -(A + C)

Information model:
  * "own"   - the player knows only their own 5 cards and the upcard (a real casino).
  * "table" - the player also sees every other player's hand (full card sharing).

Speed: every 5-card hand is ranked once with `poker.hand_rank` into an integer
score table indexed by the combinatorial number system, so a decision over all
C(46,4) = 163,185 dealer completions is a vectorised numpy lookup.
"""
from __future__ import annotations

import os
from itertools import combinations
from math import comb

import numpy as np

from .poker import hand_rank

RANKS = "23456789TJQKA"
SUITS = "hdcs"
DECK = [r + s for r in RANKS for s in SUITS]          # card id = 4*rank_index + suit_index
CARD_ID = {c: i for i, c in enumerate(DECK)}

MULTIPLIERS = np.array([1, 1, 2, 3, 4, 5, 7, 20, 50, 100])  # by hand category 0..9
N_HANDS = comb(52, 5)
_BASE = 15                                            # tiebreaker values are 2..14

# _BINOM[n, k] = C(n, k); used to index a sorted 5-card hand c0<c1<..<c4 as
# sum_k C(c_k, k+1) (combinatorial number system, unique in [0, C(52,5))).
_BINOM = np.array([[comb(n, k) for k in range(6)] for n in range(53)], dtype=np.int64)

_CACHE = os.path.join(os.path.dirname(__file__), ".cache_hand_scores.npz")


def score_from_rank(rank: tuple) -> int:
    """Map hand_rank's (category, tiebreakers) to one integer with the same order."""
    cat, tb = rank
    tb = list(tb) + [0] * (5 - len(tb))
    s = cat
    for v in tb:
        s = s * _BASE + v
    return s


def _category_from_score(score: np.ndarray) -> np.ndarray:
    return score // _BASE ** 5


def hand_index(sorted_ids: np.ndarray) -> np.ndarray:
    """Index of sorted 5-card hands (shape (..., 5), ascending card ids)."""
    idx = np.zeros(sorted_ids.shape[:-1], dtype=np.int64)
    for k in range(5):
        idx += _BINOM[sorted_ids[..., k], k + 1]
    return idx


def build_tables(use_cache: bool = True):
    """Return (score, qualifies) arrays over all C(52,5) hands."""
    if use_cache and os.path.exists(_CACHE):
        z = np.load(_CACHE)
        return z["score"], z["qualifies"]
    score = np.empty(N_HANDS, dtype=np.int64)
    qual = np.empty(N_HANDS, dtype=bool)
    for ids in combinations(range(52), 5):
        r = hand_rank([DECK[i] for i in ids])
        i = sum(comb(c, k + 1) for k, c in enumerate(ids))
        score[i] = score_from_rank(r)
        cat, tb = r
        qual[i] = cat >= 1 or (tb[0] == 14 and tb[1] == 13)   # pair+ or A-K high
    if use_cache:
        np.savez(_CACHE, score=score, qualifies=qual)
    return score, qual


_SCORE = None
_QUAL = None
_COMBOS: dict[int, np.ndarray] = {}


def _tables():
    global _SCORE, _QUAL
    if _SCORE is None:
        _SCORE, _QUAL = build_tables()
    return _SCORE, _QUAL


def _combos4(n: int) -> np.ndarray:
    if n not in _COMBOS:
        _COMBOS[n] = np.array(list(combinations(range(n), 4)), dtype=np.int64)
    return _COMBOS[n]


def to_ids(cards) -> list[int]:
    return [c if isinstance(c, (int, np.integer)) else CARD_ID[c] for c in cards]


def hand_score(cards) -> int:
    score, _ = _tables()
    ids = np.sort(np.array(to_ids(cards)))
    return int(score[hand_index(ids)])


def call_ev(player, upcard, dead=(), ante: float = 1.0, call: float = 2.0) -> dict:
    """Exact EV of calling, averaged over every dealer completion consistent with
    the known cards (player hand + upcard + `dead`, e.g. other players' hands)."""
    score, qual = _tables()
    known = set(to_ids(player)) | {to_ids([upcard])[0]} | set(to_ids(dead))
    unseen = np.array([c for c in range(52) if c not in known], dtype=np.int64)
    combos = unseen[_combos4(len(unseen))]
    up = to_ids([upcard])[0]
    dealer = np.sort(np.concatenate([np.full((len(combos), 1), up), combos], axis=1), axis=1)
    idx = hand_index(dealer)
    d_score, d_qual = score[idx], qual[idx]

    p_score = hand_score(player)
    mult = MULTIPLIERS[_category_from_score(np.int64(p_score))]
    n = len(idx)
    p_nq = np.count_nonzero(~d_qual) / n
    q_score = d_score[d_qual]
    p_win = np.count_nonzero(p_score > q_score) / n
    p_tie = np.count_nonzero(p_score == q_score) / n
    p_lose = np.count_nonzero(p_score < q_score) / n
    ev = p_nq * ante + p_win * (ante + call * mult) + p_lose * -(ante + call)
    return {"ev_call": ev, "ev_fold": -ante, "call": ev > -ante,
            "p_not_qualified": p_nq, "p_win": p_win, "p_tie": p_tie, "p_lose": p_lose,
            "multiplier": int(mult), "n_dealer_hands": n}


def settle(player, dealer, called: bool, ante: float = 1.0, call: float = 2.0) -> float:
    """Net P&L of one hand against the dealer's actual 5 cards."""
    if not called:
        return -ante
    score, qual = _tables()
    d_idx = hand_index(np.sort(np.array(to_ids(dealer))))
    if not qual[d_idx]:
        return ante
    p, d = hand_score(player), int(score[d_idx])
    if p > d:
        return ante + call * int(MULTIPLIERS[_category_from_score(np.int64(p))])
    if p == d:
        return 0.0
    return -(ante + call)


def simulate_table(num_players: int = 7, num_rounds: int = 500, info: str = "own",
                   seed: int = 0, ante: float = 1.0, call: float = 2.0) -> dict:
    """Play `num_rounds` rounds at one table. One shuffled deck and ONE dealer hand
    per round, shared by every player. Each player calls iff EV(call) > -ante under
    the chosen information model.

    Returns per-round, per-player arrays: pnl, called, ev (EV of the chosen action).
    """
    if info not in ("own", "table"):
        raise ValueError("info must be 'own' or 'table'")
    rng = np.random.default_rng(seed)
    pnl = np.zeros((num_rounds, num_players))
    called = np.zeros((num_rounds, num_players), dtype=bool)
    ev = np.zeros((num_rounds, num_players))
    for r in range(num_rounds):
        deck = rng.permutation(52)
        hands = [deck[5 * i:5 * i + 5] for i in range(num_players)]
        pos = 5 * num_players
        upcard = int(deck[pos])
        dealer = np.concatenate([[upcard], deck[pos + 1:pos + 5]])
        for i, h in enumerate(hands):
            dead = ([c for j, o in enumerate(hands) if j != i for c in o]
                    if info == "table" else ())
            d = call_ev(h, upcard, dead, ante, call)
            called[r, i] = d["call"]
            ev[r, i] = max(d["ev_call"], -ante)
            pnl[r, i] = settle(h, dealer, d["call"], ante, call)
    return {"pnl": pnl, "called": called, "ev": ev, "info": info,
            "num_players": num_players, "num_rounds": num_rounds, "seed": seed}


def summarize(values: np.ndarray, ante: float = 1.0, z: float = 1.96) -> dict:
    """Mean per hand with a 95% CI, clustered by round: every player at a table
    faces the same dealer hand, so a round (row) is the independent unit."""
    per_round = values.mean(axis=1)
    m = per_round.mean()
    se = per_round.std(ddof=1) / np.sqrt(len(per_round))
    return {"mean_per_hand": m, "se": se, "ci": (m - z * se, m + z * se),
            "house_edge_pct_of_ante": -100 * m / ante, "n_rounds": len(per_round),
            "n_hands": values.size}
