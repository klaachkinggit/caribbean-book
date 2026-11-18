"""Checks for helpers/caribbean.py against the original slow implementation."""
import random
from itertools import combinations

import numpy as np
import pytest

from helpers import caribbean as cb
from helpers.poker import compare_hands, hand_rank


def _qualifies(hand):
    r = hand_rank(hand)
    return r[0] >= 1 or (r[1][0] == 14 and r[1][1] == 13)


def _slow_net(player, dealer, ante=1, call=2):
    """Reference payout, written straight from the rules."""
    if not _qualifies(dealer):
        return ante
    res = compare_hands(player, dealer)
    if res == 1:
        return ante + call * int(cb.MULTIPLIERS[hand_rank(player)[0]])
    return 0 if res == 0 else -(ante + call)


# ---- payouts ---------------------------------------------------------------
def test_dealer_not_qualified_pays_ante_only():
    # dealer Q-high: does not qualify -> +1 (ante 1:1, call returned), not +2
    assert cb.settle(["As", "Ad", "7c", "4h", "2s"], ["Qh", "Jd", "9c", "6s", "3d"], True) == 1


def test_win_pays_ante_plus_call_times_multiplier():
    # two pair (x2) beats a qualifying pair: +1 + 2*2 = +5, not 2 + 2*2 = 6
    assert cb.settle(["Ks", "Kd", "7c", "7h", "2s"], ["Qh", "Qd", "9c", "6s", "3d"], True) == 5


def test_loss_tie_fold():
    assert cb.settle(["2s", "2d", "7c", "4h", "3s"], ["Qh", "Qd", "9c", "6s", "3d"], True) == -3
    assert cb.settle(["As", "Kd", "Qc", "Jh", "9s"], ["Ah", "Kc", "Qd", "Js", "9h"], True) == 0
    assert cb.settle(["As", "Kd", "Qc", "Jh", "9s"], ["Ah", "Kc", "Qd", "Js", "9h"], False) == -1


# ---- hand scoring ----------------------------------------------------------
def test_score_order_matches_compare_hands():
    rng = random.Random(0)
    for _ in range(3000):
        cards = rng.sample(cb.DECK, 10)
        a, b = cards[:5], cards[5:]
        sa, sb = cb.hand_score(a), cb.hand_score(b)
        assert np.sign(sa - sb) == compare_hands(a, b)


def test_hand_index_is_a_bijection_on_a_sample():
    rng = np.random.default_rng(1)
    hands = {tuple(sorted(rng.choice(52, 5, replace=False))) for _ in range(5000)}
    idx = cb.hand_index(np.array(sorted(hands)))
    assert len(set(idx.tolist())) == len(hands)
    assert idx.min() >= 0 and idx.max() < cb.N_HANDS


# ---- EV is the average of the real payout -----------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_call_ev_equals_brute_force_average(seed):
    rng = random.Random(seed)
    deck = cb.DECK[:]
    rng.shuffle(deck)
    player, up, dead = deck[:5], deck[5], deck[6:40]      # 12 unseen -> 495 dealer hands
    unseen = deck[40:]
    nets = [_slow_net(player, [up, *c]) for c in combinations(unseen, 4)]
    got = cb.call_ev(player, up, dead)
    assert got["n_dealer_hands"] == len(nets)
    assert got["ev_call"] == pytest.approx(sum(nets) / len(nets))


def test_information_models_enumerate_the_right_hand_counts():
    assert cb.call_ev(["As", "Ad", "7c", "4h", "2s"], "Kc")["n_dealer_hands"] == 163185  # C(46,4)
    deck = cb.DECK[:]
    random.Random(2).shuffle(deck)
    hands = [deck[5 * i:5 * i + 5] for i in range(7)]
    dead = [c for h in hands[1:] for c in h]
    assert cb.call_ev(hands[0], deck[35], dead)["n_dealer_hands"] == 1820               # C(16,4)


# ---- simulator ---------------------------------------------------------------
def test_simulation_settles_every_player_against_one_shared_dealer_hand():
    res = cb.simulate_table(num_players=3, num_rounds=40, info="own", seed=3)
    rng = np.random.default_rng(3)
    for r in range(40):
        deck = rng.permutation(52)
        dealer = np.concatenate([[deck[15]], deck[16:20]])
        for i in range(3):
            hand = deck[5 * i:5 * i + 5]
            assert res["pnl"][r, i] == cb.settle(hand, dealer, res["called"][r, i])
