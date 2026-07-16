"""Z-sets on Spark DataFrames.

A **Z-set** is a relation with an integer weight per row (``_trickle_d``: ``+1`` present, ``-1``
retraction). An update is a ``-1`` of the old full image plus a ``+1`` of the new, so deletions and
updates carry **full row images**, not key tombstones — which is what lets the builder join on any key
and propagate deletes soundly.

Retractions cancel by **full-row identity**, so the determinism contract applies to every column that
reaches a Z-set: projections must be deterministic (no ``now()``/``rand()``), rows must be groupable
(**no map-typed columns** — Spark cannot ``GROUP BY`` a map), and float-producing expressions are
suspect (recompute jitter surfaces as phantom retract/insert pairs).
"""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# The system namespace trickle-spark owns. Persisted *outputs* are clean tables — system columns exist
# only in-flight (Z-sets) and in state companions — but user columns under the prefix are rejected at
# write so the namespace stays available everywhere.
SYSTEM_PREFIX = "_trickle_"

D_COL = f"{SYSTEM_PREFIX}d"  # the Z-set weight
OP_COL = f"{SYSTEM_PREFIX}op"  # transient merge-source discriminator, never persisted


def user_columns(df: DataFrame) -> list[str]:
    return [c for c in df.columns if not c.startswith(SYSTEM_PREFIX)]


def check_no_system_columns(df: DataFrame, *, context: str) -> None:
    bad = [c for c in df.columns if c.startswith(SYSTEM_PREFIX) and c not in (D_COL,)]
    if bad:
        raise ValueError(f"{context}: column names under the reserved prefix {SYSTEM_PREFIX!r}: {bad}")


def as_zset(df: DataFrame, weight: int = 1) -> DataFrame:
    """Tag a plain relation as a Z-set with a constant weight."""
    return df.withColumn(D_COL, F.lit(weight).cast("long"))


def consolidate(zset: DataFrame) -> DataFrame:
    """Collapse a Z-set to net weights: full-row GROUP BY, drop rows whose weights cancel.

    Multi-step changes over a window (a→b→c→d) collapse to the net change; an unchanged row nets to 0
    and disappears. Column references are backtick-quoted because the builder's internal frames carry
    dotted ``alias.col`` names (a bare string would parse as struct access).
    """
    cols = [F.col(f"`{c}`") for c in zset.columns if c != D_COL]
    return zset.groupBy(*cols).agg(F.sum(D_COL).alias(D_COL)).where(F.col(D_COL) != 0)


def diff(new: DataFrame, old: DataFrame) -> DataFrame:
    """The consolidated Z-set turning ``old`` into ``new`` (new ``+1`` ⊎ old ``-1``).

    This is the anti-cascade primitive: a comprehensive recompute is *diffed against the current
    output* and only the real changes are applied, so the output's change feed — the next consumer's
    window — stays small even when this run recomputed everything.
    """
    if set(new.columns) != set(old.columns):
        raise ValueError(
            f"the recomputed output's schema {sorted(new.columns)} no longer matches the existing table's "
            f"{sorted(old.columns)} — trickle-spark does not evolve an output's schema in place; write the "
            f"new shape to a new table (or drop and re-bootstrap this one)"
        )
    return consolidate(as_zset(new, 1).unionByName(as_zset(old, -1)))


@dataclass(frozen=True)
class Delta:
    """A source's change over a watermark window, as a Z-set.

    ``is_full`` means there is no usable window (bootstrap, retention expiry, a recreated table, or a
    change fraction past ``p``) and ``zset`` is the source's **entire current state** as ``+1`` rows —
    the consumer must treat it comprehensively, not as an increment.
    """

    zset: DataFrame
    is_full: bool = False

    @property
    def upserts(self) -> DataFrame:
        return self.zset.where(F.col(D_COL) > 0).drop(D_COL)

    @property
    def deletes(self) -> DataFrame:
        return self.zset.where(F.col(D_COL) < 0).drop(D_COL)

    def is_empty(self) -> bool:
        return len(self.zset.take(1)) == 0
