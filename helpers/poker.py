from collections import Counter

RANK_ORDER = "23456789TJQKA"
VALUE_MAP = {r: i for i, r in enumerate(RANK_ORDER, start=2)}


def _sorted_values(cards):
    """Return card ranks sorted descending according to VALUE_MAP."""
    return sorted((VALUE_MAP[c[0]] for c in cards), reverse=True)


def _is_flush(cards):
    suits = [c[1] for c in cards]
    return len(set(suits)) == 1


def _is_straight(values):
    # Handle wheel straight (A-2-3-4-5)
    if values == [14, 5, 4, 3, 2]:
        return True, 5  # low straight, high card = 5
    for i in range(4):
        if values[i] - values[i + 1] != 1:
            return False, None
    return True, values[0]


def hand_rank(cards):
    """Return a tuple (category, tiebreakers list) for a 5-card hand."""
    values = _sorted_values(cards)
    counts = Counter(values)
    groups = sorted(((cnt, val) for val, cnt in counts.items()), reverse=True)
    flush = _is_flush(cards)
    straight, high_straight = _is_straight(values)

    # Straight / Flush / Straight Flush / Royal Flush
    if straight and flush:
        if high_straight == 14:  # 10-A straight flush
            rank_cat = 9  # Royal Flush
        else:
            rank_cat = 8  # Straight Flush
        return (rank_cat, [high_straight])

    # Four of a Kind
    if groups[0][0] == 4:
        quad_val = groups[0][1]
        kicker = groups[1][1]
        return (7, [quad_val, kicker])

    # Full House
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, [groups[0][1], groups[1][1]])

    # Flush
    if flush:
        return (5, values)

    # Straight
    if straight:
        return (4, [high_straight])

    # Three of a Kind
    if groups[0][0] == 3:
        remaining = sorted((v for v in values if v != groups[0][1]), reverse=True)
        return (3, [groups[0][1]] + remaining)

    # Two Pair
    if groups[0][0] == 2 and groups[1][0] == 2:
        pair_vals = sorted((groups[0][1], groups[1][1]), reverse=True)
        kicker = groups[2][1]
        return (2, pair_vals + [kicker])

    # One Pair
    if groups[0][0] == 2:
        pair_val = groups[0][1]
        kickers = sorted((v for v in values if v != pair_val), reverse=True)
        return (1, [pair_val] + kickers)

    # High Card
    return (0, values)


def compare_hands(hand1, hand2):
    """Compare deux mains et renvoie 1 si hand1 gagne, -1 si hand2 gagne, 0 sinon."""
    rank1 = hand_rank(hand1)
    rank2 = hand_rank(hand2)
    if rank1 > rank2:
        return 1
    if rank1 < rank2:
        return -1
    return 0


# Exemple d'utilisation rapide
if __name__ == "__main__":
    h1 = ["Ah", "2s", "As", "Jh", "Th"]  # Royal Flush
    h2 = ["9c", "6d", "8h", "6s", "2d"]  # Four of a Kind
    res = compare_hands(h1, h2)
    print("Résultat (1 = h1 gagne):", res) 