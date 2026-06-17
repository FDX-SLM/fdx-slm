"""Data mixing for SFT.

**Single/multi-turn mixing** — sample SFT records so the multi-turn / single-turn ratio
matches the configured target (e.g. ~2/3 multi-turn), without replacement and seeded for
reproducibility.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from slm_coach.data.schema import CanonicalRecord, Role, SFTRecord
from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)


def is_multi_turn(record: SFTRecord) -> bool:
    """Return whether an SFT record is multi-turn (>= 2 assistant turns).

    Uses the explicit ``conversation_type`` when it says ``multi_turn``; otherwise infers
    from the number of assistant turns.
    """
    if record.conversation_type == "multi_turn":
        return True
    return record.num_assistant_turns >= 2


def split_single_multi(
    sft_records: Sequence[SFTRecord],
) -> tuple[list[SFTRecord], list[SFTRecord]]:
    """Partition SFT records into ``(single_turn, multi_turn)`` lists."""
    single: list[SFTRecord] = []
    multi: list[SFTRecord] = []
    for record in sft_records:
        (multi if is_multi_turn(record) else single).append(record)
    return single, multi


def mix_single_multi(
    sft_records: Sequence[SFTRecord],
    *,
    multi_turn: float,
    single: float,
    seed: int = 42,
) -> list[SFTRecord]:
    """Sample SFT records to match a target multi/single ratio (no replacement).

    The largest feasible total ``T`` is chosen such that both pools can satisfy the ratio,
    then ``round(T * multi_turn)`` multi-turn and the remainder single-turn records are
    sampled and shuffled.

    Args:
        sft_records: All available SFT records.
        multi_turn: Target fraction of multi-turn records (0-1).
        single: Target fraction of single-turn records (0-1). Need not sum to 1 with
            ``multi_turn``; the two are normalized.
        seed: RNG seed for deterministic sampling.

    Returns:
        A shuffled list whose multi-turn fraction approximates ``multi_turn``.

    Raises:
        ValueError: If both ratios are non-positive.
    """
    total_ratio = multi_turn + single
    if total_ratio <= 0:
        raise ValueError("multi_turn + single must be positive")
    r_multi = multi_turn / total_ratio

    single_pool, multi_pool = split_single_multi(sft_records)
    rng = random.Random(seed)

    # Largest total T with enough records in each pool to honor the ratio.
    feasible = [len(single_pool) + len(multi_pool)]
    if r_multi > 0:
        feasible.append(int(len(multi_pool) / r_multi))
    if (1 - r_multi) > 0:
        feasible.append(int(len(single_pool) / (1 - r_multi)))
    total = max(0, min(feasible))

    n_multi = min(len(multi_pool), round(total * r_multi))
    n_single = min(len(single_pool), total - n_multi)

    chosen = rng.sample(multi_pool, n_multi) + rng.sample(single_pool, n_single)
    rng.shuffle(chosen)
    logger.info(
        "Mixed SFT records",
        extra={"n_multi": n_multi, "n_single": n_single, "target_multi_ratio": round(r_multi, 3)},
    )
    return chosen


def count_modes(records: Sequence[CanonicalRecord]) -> dict[str, int]:
    """Return a ``mode -> count`` histogram for a record collection."""
    counts: dict[str, int] = {}
    for record in records:
        counts[record.mode] = counts.get(record.mode, 0) + 1
    return counts


def assistant_turn_total(records: Sequence[CanonicalRecord]) -> int:
    """Total assistant turns across SFT records (a rough trainable-signal proxy)."""
    return sum(
        sum(1 for m in r.messages if m.role == Role.assistant)
        for r in records
        if isinstance(r, SFTRecord)
    )
