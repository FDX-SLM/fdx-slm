"""Read, validate, filter, and split the delivered JSONL into canonical records.

Directory layout consumed (``docs/SPEC.md`` §3)::

    data/sft/*.jsonl
    data/reasoning/*.jsonl
    data/preference/*.jsonl
    data/gold/gold_test.jsonl

Only records whose ``audit_status`` is in the configured keep-list (default ``approved``)
are returned for training.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from slm_coach.data.schema import (
    CanonicalRecord,
    GoldCase,
    ValidationStats,
    iter_jsonl,
    parse_gold_case,
    parse_record,
)
from slm_coach.utils.logging import get_logger

logger = get_logger(__name__)

#: Training data types loaded by default (the gold set is loaded separately for eval).
DEFAULT_DATA_TYPES: tuple[str, ...] = ("sft", "reasoning", "preference")
DEFAULT_KEEP_AUDIT: tuple[str, ...] = ("approved",)


@dataclass
class SplitResult:
    """Outcome of loading a single data-type split."""

    data_type: str
    records: list[CanonicalRecord] = field(default_factory=list)
    stats: ValidationStats = field(default_factory=ValidationStats)
    filtered_out: int = 0  # valid records dropped by the audit-status filter


@dataclass
class LoadedData:
    """All training splits loaded from a data directory."""

    splits: dict[str, SplitResult] = field(default_factory=dict)

    def __getitem__(self, data_type: str) -> list[CanonicalRecord]:
        """Return the kept records for ``data_type`` (empty list if absent)."""
        split = self.splits.get(data_type)
        return split.records if split else []

    def count(self, data_type: str) -> int:
        """Number of kept records for ``data_type``."""
        return len(self[data_type])

    @property
    def total(self) -> int:
        """Total kept records across all splits."""
        return sum(len(s.records) for s in self.splits.values())


def discover_files(data_dir: str | Path, data_type: str) -> list[Path]:
    """Return the sorted ``.jsonl`` files for one data type.

    Args:
        data_dir: Root data directory.
        data_type: Subdirectory name (e.g. ``"sft"``).

    Returns:
        Sorted list of matching file paths (possibly empty).
    """
    subdir = Path(data_dir) / data_type
    if not subdir.is_dir():
        logger.warning("Data subdirectory missing", extra={"path": str(subdir)})
        return []
    return sorted(subdir.glob("*.jsonl"))


def load_split(
    data_dir: str | Path,
    data_type: str,
    keep_audit_status: Sequence[str] | None = DEFAULT_KEEP_AUDIT,
) -> SplitResult:
    """Load, validate, and audit-filter every file for one data type.

    Args:
        data_dir: Root data directory.
        data_type: Subdirectory name (``"sft"``, ``"reasoning"``, ``"preference"``).
        keep_audit_status: Allowed ``audit_status`` values; ``None`` keeps everything.

    Returns:
        A :class:`SplitResult` with kept records, validation stats, and filtered count.
    """
    result = SplitResult(data_type=data_type)
    keep = set(keep_audit_status) if keep_audit_status is not None else None

    for path in discover_files(data_dir, data_type):
        for lineno, obj in iter_jsonl(path):
            result.stats.total += 1
            try:
                record = parse_record(obj)
            except ValidationError as exc:
                result.stats.invalid += 1
                first = exc.errors()[0]
                loc = ".".join(str(p) for p in first.get("loc", ()))
                result.stats.errors.append((lineno, f"{path.name}:{loc}: {first.get('msg')}"))
                continue
            result.stats.valid += 1
            result.stats.by_data_type[record.data_type] += 1
            result.stats.by_mode[record.mode] += 1
            if keep is not None and record.audit_status not in keep:
                result.filtered_out += 1
                continue
            result.records.append(record)

    logger.info(
        "Loaded split",
        extra={
            "data_type": data_type,
            "valid": result.stats.valid,
            "invalid": result.stats.invalid,
            "kept": len(result.records),
            "filtered_out": result.filtered_out,
        },
    )
    return result


def load_records(
    data_dir: str | Path,
    data_types: Iterable[str] = DEFAULT_DATA_TYPES,
    keep_audit_status: Sequence[str] | None = DEFAULT_KEEP_AUDIT,
) -> LoadedData:
    """Load every requested split from a data directory.

    Args:
        data_dir: Root data directory.
        data_types: Which splits to load.
        keep_audit_status: Allowed ``audit_status`` values; ``None`` keeps everything.

    Returns:
        A :class:`LoadedData` keyed by data type.
    """
    loaded = LoadedData()
    for data_type in data_types:
        loaded.splits[data_type] = load_split(data_dir, data_type, keep_audit_status)
    return loaded


def filter_by_audit(
    records: Iterable[CanonicalRecord],
    keep_audit_status: Sequence[str],
) -> list[CanonicalRecord]:
    """Return only records whose ``audit_status`` is in the keep-list."""
    keep = set(keep_audit_status)
    return [r for r in records if r.audit_status in keep]


def validate_data_dir(
    data_dir: str | Path,
    data_types: Iterable[str] = DEFAULT_DATA_TYPES,
) -> dict[str, ValidationStats]:
    """Validate every split without filtering (used by ``scripts/validate_data.py``).

    Args:
        data_dir: Root data directory.
        data_types: Which splits to validate.

    Returns:
        Mapping of data type to its :class:`ValidationStats`.
    """
    out: dict[str, ValidationStats] = {}
    for data_type in data_types:
        split = load_split(data_dir, data_type, keep_audit_status=None)
        out[data_type] = split.stats
    return out


def load_jsonl_records(
    path: str | Path,
    keep_audit_status: Sequence[str] | None = DEFAULT_KEEP_AUDIT,
) -> list[CanonicalRecord]:
    """Load, validate, and audit-filter canonical records from a single JSONL file.

    Used to read a materialized holdout split (``data/holdout/{train,val}.jsonl``). Invalid lines
    are skipped with a warning rather than aborting the load.

    Args:
        path: Path to a JSONL file of canonical records.
        keep_audit_status: Allowed ``audit_status`` values; ``None`` keeps everything.

    Returns:
        The kept :class:`CanonicalRecord` objects in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Record file not found: {file_path}")
    keep = set(keep_audit_status) if keep_audit_status is not None else None
    records: list[CanonicalRecord] = []
    for lineno, obj in iter_jsonl(file_path):
        try:
            record = parse_record(obj)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            logger.warning(
                "Skipping invalid record",
                extra={
                    "path": file_path.name,
                    "line": lineno,
                    "error": f"{loc}: {first.get('msg')}",
                },
            )
            continue
        if keep is not None and record.audit_status not in keep:
            continue
        records.append(record)
    return records


def load_gold(path: str | Path) -> list[dict]:
    """Load the gold test set as raw decoded dicts (one per line).

    Args:
        path: Path to ``gold_test.jsonl``.

    Returns:
        A list of decoded gold-case objects (unvalidated).

    Raises:
        FileNotFoundError: If the gold file does not exist.
    """
    gold_path = Path(path)
    if not gold_path.is_file():
        raise FileNotFoundError(f"Gold test set not found: {gold_path}")
    return [obj for _, obj in iter_jsonl(gold_path)]


def load_gold_cases(path: str | Path) -> list[GoldCase]:
    """Load and validate the gold test set into canonical :class:`GoldCase` objects.

    Each case is normalized to a uniform prompt (``messages``) + ``reference`` + ``mode``
    (see :class:`slm_coach.data.schema.GoldCase`).

    Args:
        path: Path to ``gold_test.jsonl``.

    Returns:
        Validated gold cases in file order.

    Raises:
        FileNotFoundError: If the gold file does not exist.
        pydantic.ValidationError: If a case violates the gold schema.
    """
    return [parse_gold_case(obj) for obj in load_gold(path)]
