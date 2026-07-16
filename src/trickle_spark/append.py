"""Append-only outputs: the ``.append_to()`` terminal's runner and its conflict semantics.

An **append-only** output is a Delta table that only ever receives inserts — for a *monotonic*
transform (output rows only added, never updated or retracted; e.g. an append-only fact stream joined
to stable dims). Its CDF is inserts-only, its history is immutable, and the per-run commit is a plain
``append`` carrying the watermarks (or the heartbeat when nothing new arrived).

An insert-only table can't reflect a *change to the past*, so two things are **conflicts** (ported from
the reference ``trickle/io.py:append_zset``):

- a **retraction** (a ``-1`` row in the composed ΔO) — a previously-emitted output row changed or
  disappeared upstream;
- a present (``+1``) row whose ``pk`` is already in the table with a **different** image.

A ``+1`` row whose ``pk`` is already present with an **identical** image is a benign skip (an
idempotent replay or a comprehensive re-derivation re-producing it) — never a conflict, never logged.
A ``pk`` duplicated *within one run* with distinct images is unresolvable (no recency to choose by) and
always raises. ``fail_on_conflict=True`` (default — correctness over speed) raises before writing
anything; ``False`` drops the conflicting rows (history wins, the past stays frozen) and appends the
rest — with ``log_drops`` the dropped rows land in a ``{output}__trickle_droplog`` companion Delta
table (the user columns + the ``_trickle_d`` sign), an append-only diagnostic. ``pk=()`` skips the pk
checks entirely (only retractions conflict) — fast, sound only when duplicates and past-changes are
impossible by construction.

**Replay safety**: the output append is one commit carrying the pins, exactly like a merge output. The
droplog is a second table, so it is written **first, pins-guarded** (the same order-and-fast-forward
treatment as the agg/acc state companions): a crash between the droplog write and the output commit
re-plans the same window, re-derives the same drops, sees the droplog already at these pins, and skips
straight to the append.

A **comprehensive** run (bootstrap / coverage-miss / over-``p``) is the whole recomputed output tagged
``+1`` and *append-filtered* against history — re-derived identical rows skip benignly, changed ones
conflict — so fallbacks never spuriously fail, and never rewrite the past.
"""

from __future__ import annotations

from typing import Callable

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .apply import heartbeat, write_bootstrap
from .run import RunContext, RunResult, pin_of
from .tables import Pin, commit_metadata, commit_metadata_json, read_watermarks, table_exists
from .zset import D_COL, as_zset, check_no_system_columns, consolidate

DROPLOG_SUFFIX = "__trickle_droplog"


class AppendConflict(ValueError):
    """The composed change is not append-safe (a retraction, a changed image for an existing ``pk``,
    or a within-run ``pk`` duplicate)."""


def droplog_for(output: str) -> str:
    return f"{output}{DROPLOG_SUFFIX}"


def run_append(
    spark: SparkSession,
    output: str,
    *,
    sources: list[str],
    pk: tuple[str, ...],
    full: Callable[[RunContext], DataFrame],
    delta: Callable[[RunContext], DataFrame | None] | None = None,
    p: float | dict[str, float] | None = None,
    fail_on_conflict: bool = True,
    log_drops: bool = True,
    tag: str | None = None,
) -> RunResult:
    """One maintenance step for an **append-only** ``output`` — :func:`~.run.run`'s sibling with the
    same planning ladder (skip / bootstrap / incremental / comprehensive, one watermark-carrying commit)
    but an append apply instead of a MERGE: a comprehensive result is tagged ``+1`` and append-filtered
    against history rather than diffed. See the module docstring for the conflict semantics."""
    if not sources:
        raise ValueError("run_append() needs at least one source")
    p_map = dict(p) if isinstance(p, dict) else {s: p for s in sources} if p is not None else {}

    exists = table_exists(spark, output)
    last = read_watermarks(spark, output) if exists else None
    pins = {s: pin_of(spark, s) for s in sources}

    if exists and last is not None and all(last.get(s) == pins[s] for s in sources):
        return RunResult(status="skipped", pins=pins, changed=False)

    ctx = RunContext(spark=spark, pins=pins, last=last, p=p_map)

    if not exists:
        appended = _apply_append(
            spark, output, as_zset(full(ctx), 1), pk, "bootstrap", pins,
            fail_on_conflict=fail_on_conflict, log_drops=log_drops, bootstrap=True, tag=tag,
        )
        return RunResult(status="bootstrap", pins=pins, changed=appended)

    zset = None
    status = "comprehensive"
    if last is not None and delta is not None:
        zset = delta(ctx)
        status = "incremental" if zset is not None else "comprehensive"
    if zset is None:
        zset = as_zset(full(ctx), 1)
    appended = _apply_append(
        spark, output, zset, pk, status, pins,
        fail_on_conflict=fail_on_conflict, log_drops=log_drops, bootstrap=False, tag=tag,
    )
    return RunResult(status=status, pins=pins, changed=appended)


def _apply_append(
    spark: SparkSession,
    output: str,
    zset: DataFrame,
    pk: tuple[str, ...],
    kind: str,
    pins: dict[str, Pin],
    *,
    fail_on_conflict: bool,
    log_drops: bool,
    bootstrap: bool,
    tag: str | None = None,
) -> bool:
    """Consolidate, vet the conflicts, and land the new rows as one append commit (or the heartbeat).
    Returns whether any genuinely new rows were appended (benign skips and dropped conflicts don't
    count)."""
    z = consolidate(zset).cache()
    check_no_system_columns(z, context=f"append to {output}")
    user = [c for c in z.columns if c != D_COL]
    if pk:
        missing = [c for c in pk if c not in user]
        if missing:
            raise ValueError(f"append to '{output}': primary key column(s) {missing} not in the composed output")
    metadata_json = commit_metadata_json(kind, pins, tag=tag)
    plus = z.where(F.col(D_COL) > 0).drop(D_COL)
    retractions = z.where(F.col(D_COL) < 0).drop(D_COL)

    if pk and plus.groupBy(*[F.col(f"`{c}`") for c in pk]).count().where("count > 1").take(1):
        # Two distinct images for one key in one run — no recency to disambiguate; always fatal.
        raise AppendConflict(
            f"append to '{output}': primary key {pk} has duplicate key(s) with differing values in one "
            f"run — the output is not unique by {pk}"
        )

    conflicts = None
    if pk and not bootstrap:
        # +1 rows whose pk exists in history with a *different* image (identical images skip benignly).
        eq = " AND ".join(f"p.`{c}` <=> h.`{c}`" for c in user)
        on = " AND ".join(f"p.`{c}` = h.`{c}`" for c in pk)
        conflicts = (
            plus.alias("p")
            .join(spark.table(output).alias("h"), on=F.expr(on), how="inner")
            .where(f"NOT ({eq})")
            .select(*[F.col(f"p.`{c}`").alias(c) for c in user])
        )
    has_retraction = bool(retractions.take(1))
    has_conflict = bool(conflicts is not None and conflicts.take(1))
    if fail_on_conflict and (has_retraction or has_conflict):
        raise AppendConflict(
            f"append to '{output}': not append-safe — the change carries "
            f"{'retraction(s)' if has_retraction else ''}{' and ' if has_retraction and has_conflict else ''}"
            f"{'changed-past row(s)' if has_conflict else ''}. Pass fail_on_conflict=False to drop them, "
            f"or use .merge_into() to track changes."
        )

    if pk and not bootstrap:
        new_rows = plus.join(spark.table(output).select(*[F.col(f"`{c}`") for c in pk]), on=list(pk), how="left_anti")
    else:
        new_rows = plus
    # Eager and lineage-truncating: the append below writes to the very table new_rows' anti-join reads.
    new_rows = new_rows.localCheckpoint()

    # Droplog first, pins-guarded (the state-companion commit order): a crash between the two commits
    # re-plans the same window, re-derives the same drops, and skips the re-log.
    if log_drops and not fail_on_conflict and (has_retraction or has_conflict):
        _log_drops(spark, output, retractions, conflicts, pins)

    if bootstrap:
        write_bootstrap(spark, output, new_rows, metadata_json)
        return bool(new_rows.take(1))
    if not new_rows.take(1):
        heartbeat(spark, output, metadata_json)
        return False
    with commit_metadata(spark, metadata_json):
        new_rows.write.format("delta").mode("append").saveAsTable(output)
    return True


def _log_drops(spark: SparkSession, output: str, retractions: DataFrame, conflicts: DataFrame | None,
               pins: dict[str, Pin]) -> None:
    """Record what ``fail_on_conflict=False`` dropped — retractions (``_trickle_d`` = −1) and
    changed-image collisions (+1) — in the ``{output}__trickle_droplog`` companion, one append commit
    stamped with the run's pins (the replay guard)."""
    drop_table = droplog_for(output)
    dropped = retractions.withColumn(D_COL, F.lit(-1).cast("long"))
    if conflicts is not None:
        dropped = dropped.unionByName(conflicts.withColumn(D_COL, F.lit(1).cast("long")))
    dropped = dropped.localCheckpoint()
    if not dropped.take(1):
        return
    if read_watermarks(spark, drop_table) == pins:
        return  # a crashed run already logged this window — skip straight to the output append
    with commit_metadata(spark, commit_metadata_json("droplog", pins)):
        dropped.write.format("delta").mode("append").saveAsTable(drop_table)
