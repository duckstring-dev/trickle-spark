"""The change source: a source table's Z-set window between two pins.

Delta **Change Data Feed** is the changelog — a ``MERGE`` into a Delta table emits row-level
``insert`` / ``delete`` / ``update_preimage`` / ``update_postimage`` rows per commit, which map exactly
to a full-row Z-set (pre-image → ``-1``, post-image/insert → ``+1``, delete → ``-1``). Nothing is
maintained by hand; every CDF-enabled table participates automatically.

``delta_of`` is the **change-source seam**: everything downstream consumes a :class:`~.zset.Delta`, so
an explicit-changelog backend (user-controlled epochs, cross-system draws) can be added later without
touching the join engine.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.utils import AnalysisException

from .tables import Pin, read_as_of, tagged_versions
from .zset import D_COL, Delta, as_zset, consolidate

_CDF_COLS = ("_change_type", "_commit_version", "_commit_timestamp")
_POSITIVE = ("insert", "update_postimage")


def _full(spark: SparkSession, source: str, pin: Pin) -> Delta:
    return Delta(zset=as_zset(read_as_of(spark, source, pin.version)), is_full=True)


def _cdf_window(spark: SparkSession, source: str, lo: int, hi: int) -> DataFrame:
    df = (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", lo)
        .option("endingVersion", hi)
        .table(source)
    )
    _ = df.schema  # force analysis so retention/CDF-availability errors surface here, not at first action
    return df


def changes_at_tag(spark: SparkSession, table: str, tag: str) -> Delta:
    """The consolidated Z-set the run(s) stamped ``tag`` applied to ``table`` — **content-addressed**
    by the tag, so no consumer-side watermark is needed anywhere: a caller that tags each run with its
    own epoch (``merge_into(..., tag=f)``) recovers exactly that epoch's change on demand, and a
    replay recovers the identical window. Empty (never ``is_full``) when no commit carries the tag
    (the run skipped) or the tagged commit was a heartbeat."""
    versions = tagged_versions(spark, table, tag)
    if not versions:
        return Delta(zset=as_zset(spark.table(table)).limit(0), is_full=False)
    cdf = _cdf_window(spark, table, versions[0], versions[-1])
    cdf = cdf.where(F.col("_commit_version").isin(versions))  # untagged commits in the range don't count
    zset = consolidate(
        cdf.withColumn(D_COL, F.when(F.col("_change_type").isin(*_POSITIVE), 1).otherwise(-1).cast("long")).drop(*_CDF_COLS)
    )
    return Delta(zset=zset, is_full=False)


def delta_of(spark: SparkSession, source: str, pin: Pin, last: Pin | None, *, p: float | None = None) -> Delta:
    """The source's consolidated Z-set change over ``(last.version, pin.version]``.

    Falls back to a **full** read (``is_full=True``, entire current state as ``+1``) when the window is
    unusable: no watermark (bootstrap), a recreated table (``table_id`` mismatch), an unreadable CDF
    range (retention expiry, CDF enabled after the watermark), or — with ``p`` set — a consolidated
    change touching more than ``p`` of the source's current rows, past which incremental maintenance
    loses to recomputation. ``p=1.0`` (or ``None``) disables the fraction check.

    Retention expiring is a lag SLA, not a correctness cliff: every rung lands on the same
    comprehensive fallback the consumer must already handle for bootstrap.
    """
    if last is None or last.table_id != pin.table_id:
        return _full(spark, source, pin)
    if last.version >= pin.version:
        empty = as_zset(read_as_of(spark, source, pin.version)).limit(0)
        return Delta(zset=empty, is_full=False)
    try:
        cdf = _cdf_window(spark, source, last.version + 1, pin.version)
    except AnalysisException:
        return _full(spark, source, pin)
    zset = consolidate(
        cdf.withColumn(D_COL, F.when(F.col("_change_type").isin(*_POSITIVE), 1).otherwise(-1).cast("long")).drop(*_CDF_COLS)
    )
    if p is not None and p < 1.0:
        current = read_as_of(spark, source, pin.version).count()
        if zset.count() > p * max(current, 1):
            return _full(spark, source, pin)
    return Delta(zset=zset, is_full=False)
