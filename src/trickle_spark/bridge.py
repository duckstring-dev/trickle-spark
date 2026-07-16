"""Bridging to an external incremental world: land foreign Z-sets, read windowed Z-sets back out.

trickle-spark's change axis is the Delta table version; a sibling engine's is whatever it stamps its
own changelog with (Duckstring's Trickle uses an epoch ``f``). The two never need unifying, because
the **Z-set is the lingua franca** — full-row images with ±1 weights carry everything a delta means,
and each engine's clock is just its windowing mechanism. Crossing the boundary therefore takes exactly
two engine-neutral primitives:

- :func:`apply_changes` — the **inbound sink**: apply a foreign Z-set to a Delta *landing table* in
  one MERGE commit (created CDF-on when absent). From that table on, CDF re-derives the changelog, so
  every downstream plan is fully native with zero awareness of where the changes came from. The commit
  can carry an opaque ``tag`` (the foreign epoch): re-applying the same tag is a no-op, which makes a
  replay-at-the-same-epoch upstream (Duckstring's crash-recovery model) exactly-once here too.
- :func:`table_changes` — the **outbound reader**: a table's consolidated Z-set window since a
  :class:`~.tables.Pin` the *caller* remembers (trickle-spark's own outputs keep their watermarks in
  their own commits; a foreign consumer keeps its consumed-pin wherever its replay story lives — e.g.
  a Duckstring registry table keyed by ``f``, making the window content-addressed per run).

This pair is what the design doc's reserved "explicit-changelog change source" turned out to need in
practice: landing-then-CDF covers the cross-system case without any reconstruction-based read
machinery inside the engine. A worked Duckstring↔Spark round-trip lives in
``tests/test_duckstring_bridge.py``.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .apply import apply_zset, write_bootstrap
from .changes import delta_of
from .tables import Pin, commit_metadata_json, pin_table, read_tag, table_exists
from .zset import D_COL, Delta, consolidate, diff


def apply_changes(
    spark: SparkSession,
    table: str,
    changes: DataFrame,
    pk: tuple[str, ...],
    *,
    tag: str | None = None,
    full: bool = False,
) -> bool:
    """Apply a foreign Z-set ``changes`` (user columns + ``_trickle_d``) to the landing ``table`` in
    one commit, creating it (CDF on) when absent. Returns whether any data actually changed.

    ``tag`` is an opaque idempotency token — the producer's own epoch/run id. When the table's latest
    trickle commit already carries this tag the call is a no-op (``False``): a producer that replays a
    window re-lands nothing. ``full=True`` says ``changes``' present rows are the producer's **complete
    current state** (a foreign bootstrap or coverage-miss): it is diffed against the landing table so
    only real changes commit — the same anti-cascade rule as a comprehensive run.
    """
    if not pk:
        raise ValueError(f"apply_changes('{table}'): pass the landing key, pk=...")
    if tag is not None and read_tag(spark, table) == tag:
        return False  # this epoch already landed — a replay
    metadata = commit_metadata_json("ingest", {}, tag=tag)
    if not table_exists(spark, table):
        rows = _present(changes)
        if rows.groupBy(*[F.col(f"`{c}`") for c in pk]).count().where("count > 1").take(1):
            raise ValueError(f"apply_changes('{table}'): the landing state is not unique by {pk}")
        write_bootstrap(spark, table, rows, metadata)
        return bool(spark.table(table).take(1))
    if full:
        changes = diff(_present(changes), spark.table(table))
    return apply_zset(spark, table, changes, pk, metadata) == "merged"


def _present(changes: DataFrame) -> DataFrame:
    """The net-present rows of a Z-set — or the frame itself when it carries no weight column (a
    producer handing over a plain full state)."""
    if D_COL not in changes.columns:
        return changes
    return consolidate(changes).where(f"`{D_COL}` > 0").drop(D_COL)


def table_changes(spark: SparkSession, table: str, after: Pin | None, *, p: float | None = None) -> tuple[Delta, Pin]:
    """The consolidated Z-set change of ``table`` since the pin ``after``, plus the **current pin** to
    remember for the next call — the outbound half of the bridge, for a foreign consumer that keeps
    its own consumed-watermark (trickle-spark outputs keep theirs in their own commits; a foreign one
    stores the returned pin wherever its replay story lives).

    ``after=None`` (or a recreated table, or an expired window) yields a **full** read
    (``delta.is_full`` — the entire current state as ``+1``), exactly the fallback ladder a plan's own
    sources ride; ``p`` is the usual change-fraction threshold."""
    pin = pin_table(spark, table)
    return delta_of(spark, table, pin, after, p=p), pin
