"""Mode-stratified held-out validation split.

The legacy in-memory split (``training.sft.split_holdout``) shuffles all records and slices off a
flat fraction — on a small dataset with 7 conversation modes that leaves some modes absent from
``val`` and makes ``eval_loss`` an unrepresentative signal. :func:`stratified_holdout` instead
holds out an EVEN number of records per mode so every mode is present in ``val``, sized to at least
``min_total`` records and at least ``fraction`` of the data. The split is deterministic in ``seed``
so the same val set is held out across runs (keeping LoRA vs QLoRA comparisons consistent).
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TypeVar

from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


def stratified_holdout(
    items: Sequence[T],
    *,
    mode_of: Callable[[T], str],
    fraction: float,
    min_total: int,
    seed: int,
) -> tuple[list[T], list[T]]:
    """Split ``items`` into ``(train, val)`` with every mode represented in ``val``.

    The validation target is ``max(min_total, ceil(fraction * len(items)))`` records, distributed
    as evenly as possible across the distinct modes (so each mode contributes roughly
    ``target / n_modes`` records, capped by how many that mode actually has, and always at least
    one). Within each mode the choice is a deterministic shuffle seeded by ``seed``.

    Args:
        items: All candidate records (already audit-filtered).
        mode_of: Extracts the mode label from a record.
        fraction: Minimum val size as a fraction of ``len(items)`` (0-1).
        min_total: Minimum val size in absolute records.
        seed: RNG seed for the per-mode shuffle (reproducible split).

    Returns:
        ``(train_records, val_records)`` — disjoint; both shuffled.

    Raises:
        ValueError: If ``items`` is empty, or the val target meets/exceeds the dataset size
            (no records would remain for training).
    """
    if not items:
        raise ValueError("Cannot split an empty record set")

    by_mode: dict[str, list[T]] = defaultdict(list)
    for item in items:
        by_mode[mode_of(item)].append(item)

    modes = sorted(by_mode)
    total = len(items)
    target = max(min_total, math.ceil(fraction * total))
    if target >= total:
        raise ValueError(
            f"val target ({target}) >= dataset size ({total}); add more data or lower "
            f"val_fraction/val_min_total"
        )

    rng = random.Random(seed)
    base, remainder = divmod(target, len(modes))

    train: list[T] = []
    val: list[T] = []
    per_mode: dict[str, int] = {}
    for i, mode in enumerate(modes):
        group = list(by_mode[mode])
        rng.shuffle(group)
        # Even share (+1 for the first `remainder` modes), but always ≥1 so every mode appears,
        # and ≤ len(group)-1 so the mode is not wiped out of the training set.
        want = max(1, base + (1 if i < remainder else 0))
        take = min(want, max(1, len(group) - 1)) if len(group) > 1 else len(group)
        val.extend(group[:take])
        train.extend(group[take:])
        per_mode[mode] = take

    rng.shuffle(train)
    rng.shuffle(val)
    logger.info(
        "Stratified holdout",
        extra={
            "n_train": len(train),
            "n_val": len(val),
            "target": target,
            "modes": len(modes),
            "per_mode": per_mode,
        },
    )
    return train, val
