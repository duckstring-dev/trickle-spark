"""Duckstring Pond outputs as **native** trickle-spark sources — no landing tables, no duckdb.

A Duckstring Pond publishes its tables as plain parquet (the flat data-plane layout): per-run
**parts** stamped ``_duckstring_f`` for append histories and merge ``__changelog``/``__band`` tiers,
size-bounded ``{table}__base/`` chunks for a merge main's cold base, one ``{table}.parquet`` for a
plain overwrite output, and a ``_trickle.json`` sidecar carrying ``{mode, pk, floor, f, f_base}`` per
table. That *is* an interchange format, so ``ts.duckstring_source(data_dir, table)`` reads it
directly:

- the **delta** is the changelog/history parts filtered ``_duckstring_f ∈ (previous_f, f]``,
  consolidated — duckstring's own consumer window read, in Spark;
- the **current state** is the documented reconstruct rule (cold base anti-joined by the changed
  keys ⊎ the net-present changelog images above ``f_base``);
- the **old state** needs no time travel: it is ``current ⊎ (−δ)`` consolidated (:func:`rewind`);
- the **watermark** is the source's published epoch ``f``, riding the Spark output's own commit
  metadata exactly like a Delta pin (``Pin(version=0, table_id=<f>)`` — ``f`` is an opaque monotonic
  token to that machinery), so nothing duckstring-side tracks any Spark state;
- **coverage** mirrors duckstring's consumer rules: no watermark, ``previous_f`` below the published
  ``floor`` (a refresh/retention gap), or a window past the change fraction ``p`` → a full read.

The return path is :func:`~.changes.changes_at_tag`: stamp each Spark run with the Pond run's ``f``
(``merge_into(..., tag=f)``) and the change to hand back *at* that epoch is the CDF of the commits
tagged with it — content-addressed by the epoch itself, so a duckstring replay re-derives the
identical window with no bookkeeping table anywhere.

**Format coupling, stated plainly:** this module depends on duckstring's *published* on-disk layout
(part naming, the sidecar schema, the reconstruct rule) as of duckstring ≥ 0.4 — a read-only,
documented contract, guarded by the round-trip test (``tests/test_duckstring_bridge.py``). It reads
the flat layer both data-plane backends write (the Iceberg plane keeps flat sidecars for exactly this
kind of consumer). ``data_dir`` must be a filesystem path both this process and Spark can reach (a
shared/fuse mount for a remote cluster); object-store listing is not wired yet.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .tables import Pin
from .zset import D_COL, Delta, as_zset, consolidate

REF_PREFIX = "duckstring:"
SIDECAR = "_trickle.json"
DUCK_F = "_duckstring_f"
DUCK_D = "_duckstring_d"
CHANGELOG_SUFFIX, WARM_SUFFIX, BASE_SUFFIX = "__changelog", "__band", "__base"


def duckstring_source(data_dir, table: str, *, p: float = 0.3):
    """Start a plan at a Duckstring Pond's published ``table`` under ``data_dir`` (the Pond's data
    directory holding ``_trickle.json``). Composes with Delta sources freely — each source windows by
    its own axis. ``p`` is the usual change-fraction threshold."""
    from .builder import Builder, _Source

    return Builder(_Source(ref_of(data_dir, table), p))


def ref_of(data_dir, table: str) -> str:
    return f"{REF_PREFIX}{Path(data_dir)}::{table}"


def is_duckstring_ref(ref) -> bool:
    return isinstance(ref, str) and ref.startswith(REF_PREFIX)


def _parse(ref: str) -> tuple[Path, str]:
    body = ref[len(REF_PREFIX):]
    data_dir, _, table = body.rpartition("::")
    return Path(data_dir), table


def _sidecar_entry(data_dir: Path, table: str) -> dict:
    path = data_dir / SIDECAR
    if not path.exists():
        raise FileNotFoundError(f"{data_dir}: no {SIDECAR} — not a published duckstring data dir")
    entry = json.loads(path.read_text()).get(table)
    if entry is None:
        raise FileNotFoundError(f"{data_dir}: table '{table}' is not in the {SIDECAR} sidecar")
    return entry


def _dt(iso: str | None) -> datetime | None:
    return datetime.fromisoformat(iso) if iso else None


def _parts(data_dir: Path, table: str) -> list[Path]:
    d = data_dir / table
    return sorted(d.glob("*.parquet")) if d.is_dir() else []


def pin_source(ref: str) -> Pin:
    """The source's published epoch ``f`` as this run's pin — an opaque monotonic token to the
    watermark machinery (recorded, compared for the skip check, and handed back as ``last``)."""
    data_dir, table = _parse(ref)
    f = _sidecar_entry(data_dir, table).get("f")
    if f is None:
        raise ValueError(f"{ref}: the sidecar carries no published freshness (was the Pond ever run?)")
    return Pin(version=0, table_id=f)


# ─── reads ──────────────────────────────────────────────────────────────────────


def _read_parts(spark: SparkSession, files: list[Path]) -> DataFrame:
    return spark.read.parquet(*[str(f) for f in files])


def current_state(spark: SparkSession, ref: str, pin: Pin) -> DataFrame:
    """The source's clean current state as of the pinned epoch (system columns stripped)."""
    data_dir, table = _parse(ref)
    entry = _sidecar_entry(data_dir, table)
    f = _dt(pin.table_id)
    mode = entry.get("mode", "overwrite")
    if mode == "overwrite":
        return spark.read.parquet(str(data_dir / f"{table}.parquet"))
    if mode == "append":
        hist = _read_parts(spark, _parts(data_dir, table))
        return hist.where(F.col(DUCK_F) <= F.lit(f)).drop(DUCK_F)
    return _reconstruct(spark, data_dir, table, entry, f)


def _reconstruct(spark: SparkSession, data_dir: Path, table: str, entry: dict, f: datetime) -> DataFrame:
    """A merge main's current state: latest-per-image over the changelog tiers above ``f_base``,
    overlaid on the cold base (anti-joined by every changed key) — duckstring's reconstruct rule."""
    pk = list(entry.get("pk", ()))
    f_base = _dt(entry.get("f_base"))
    clog_files = _parts(data_dir, f"{table}{CHANGELOG_SUFFIX}") + _parts(data_dir, f"{table}{WARM_SUFFIX}")
    base_files = _parts(data_dir, f"{table}{BASE_SUFFIX}")
    legacy_base = data_dir / f"{table}.parquet"

    net = None
    if clog_files:
        clog = _read_parts(spark, clog_files).where(F.col(DUCK_F) <= F.lit(f))
        if f_base is not None:
            clog = clog.where(F.col(DUCK_F) > F.lit(f_base))
        net = consolidate(clog.drop(DUCK_F).withColumnRenamed(DUCK_D, D_COL))

    base = None
    if base_files:
        base = _read_parts(spark, base_files).drop(DUCK_F)
    elif legacy_base.exists():
        base = spark.read.parquet(str(legacy_base)).drop(DUCK_F)

    if net is None:
        if base is None:
            raise FileNotFoundError(f"{data_dir}: merge table '{table}' has no base and no changelog")
        return base
    present = net.where(F.col(D_COL) > 0).drop(D_COL)
    if base is None:
        return present
    changed_keys = net.select(*[F.col(f"`{c}`") for c in pk]).distinct()
    return base.join(F.broadcast(changed_keys), on=pk, how="left_anti").unionByName(present)


def delta_of_ref(spark: SparkSession, ref: str, pin: Pin, last: Pin | None, *, p: float | None = None) -> Delta:
    """The source's consolidated Z-set window ``(last.f, pin.f]`` — or a full read (``is_full``) on
    the same ladder duckstring's own consumers ride: bootstrap, a ``floor`` above the watermark
    (refresh / retention), an overwrite source that advanced, or a window past ``p``."""
    data_dir, table = _parse(ref)
    entry = _sidecar_entry(data_dir, table)
    mode = entry.get("mode", "overwrite")
    f = _dt(pin.table_id)
    prev = _dt(last.table_id) if last is not None else None
    floor = _dt(entry.get("floor"))

    def full() -> Delta:
        return Delta(zset=as_zset(current_state(spark, ref, pin)), is_full=True)

    if mode == "overwrite":
        if prev is not None and f is not None and f <= prev:
            return Delta(zset=as_zset(current_state(spark, ref, pin)).limit(0), is_full=False)
        return full()

    files = _parts(data_dir, f"{table}{CHANGELOG_SUFFIX}") if mode == "merge" else _parts(data_dir, table)
    if mode == "merge":
        files = files + _parts(data_dir, f"{table}{WARM_SUFFIX}")
    if prev is None or not files:
        return full()
    if floor is not None and prev < floor:
        return full()  # the consumer fell behind the retained window (or the Pond was refreshed)

    window = _read_parts(spark, files).where((F.col(DUCK_F) > F.lit(prev)) & (F.col(DUCK_F) <= F.lit(f)))
    if mode == "append":
        zset = as_zset(window.drop(DUCK_F))  # history rows are all present (+1), never retracted
    else:
        zset = consolidate(window.drop(DUCK_F).withColumnRenamed(DUCK_D, D_COL))
    if p is not None and p < 1.0:
        n = zset.count()
        if n and n > p * max(current_state(spark, ref, pin).count(), 1):
            return full()
    return Delta(zset=zset, is_full=False)


def rewind(current: DataFrame, zset: DataFrame) -> DataFrame:
    """The **old state** from the current one and the window's Z-set — ``current ⊎ (−δ)``
    consolidated. No reconstruction machinery, no time travel: with the delta in hand, undoing it is
    one union."""
    undo = zset.withColumn(D_COL, -F.col(D_COL))
    return consolidate(as_zset(current).unionByName(undo)).where(F.col(D_COL) > 0).drop(D_COL)
