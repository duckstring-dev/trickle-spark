"""Applying an output delta: one atomic MERGE commit carrying the new watermarks.

The consolidated output Z-set is applied with a single ``MERGE INTO`` — net-negative keys delete,
present rows update/insert — so the data change, its CDF emission, and the watermark metadata land in
**one commit**. A crash can therefore never leave data advanced but watermarks behind (re-run sees
empty windows and skips) or vice versa (nothing committed, clean re-plan).
"""

from __future__ import annotations

import hashlib

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .tables import commit_metadata, q
from .zset import D_COL, OP_COL, SYSTEM_PREFIX, check_no_system_columns, consolidate


def _merge_condition(pk: tuple[str, ...]) -> str:
    return " AND ".join(f"t.`{k}` = s.`{k}`" for k in pk)


def apply_zset(spark: SparkSession, output: str, zset: DataFrame, pk: tuple[str, ...], metadata_json: str) -> str:
    """Apply a consolidated output Z-set to ``output`` in one commit. Returns the action taken.

    An **empty** delta still commits: the watermark must advance or every subsequent run would re-read
    and re-compute an ever-growing window for a source whose changes never reach the output. Delta
    elides data-less writes (an empty append / no-op MERGE produces **no commit**), so the vehicle is a
    metadata-only ``SET TBLPROPERTIES`` heartbeat — it always commits, carries the userMetadata, emits
    no CDF rows (downstream windows consolidate to empty and cascade skips, not work), and changes no
    data. The property value is an opaque fingerprint; the watermark read path stays userMetadata-only.
    """
    z = consolidate(zset)
    check_no_system_columns(z, context=f"apply to {output}")
    if not z.take(1):
        heartbeat(spark, output, metadata_json)
        return "empty"

    upserts = z.where(F.col(D_COL) > 0).drop(D_COL).withColumn(OP_COL, F.lit("U"))
    deletes = (
        z.where(F.col(D_COL) < 0)
        .drop(D_COL)
        .join(z.where(F.col(D_COL) > 0).select(*pk), on=list(pk), how="left_anti")
        .dropDuplicates(list(pk))
        .withColumn(OP_COL, F.lit("D"))
    )
    src = upserts.unionByName(deletes)
    cols = [c for c in src.columns if c != OP_COL]
    assign = {c: f"s.`{c}`" for c in cols}
    merge = (
        DeltaTable.forName(spark, output)
        .alias("t")
        .merge(src.alias("s"), _merge_condition(pk))
        .whenMatchedDelete(condition=f"s.`{OP_COL}` = 'D'")  # clause order matters: D wins over update
        .whenMatchedUpdate(set=assign)
        .whenNotMatchedInsert(condition=f"s.`{OP_COL}` = 'U'", values=assign)
    )
    with commit_metadata(spark, metadata_json):
        merge.execute()
    return "merged"


def heartbeat(spark: SparkSession, output: str, metadata_json: str) -> None:
    """The watermark-advance for a run that changed no data: Delta elides data-less writes, so the
    vehicle is a metadata-only ``SET TBLPROPERTIES`` commit carrying the userMetadata. It emits no CDF
    rows — downstream windows consolidate to empty and cascade skips, not work."""
    beat = hashlib.sha256(metadata_json.encode()).hexdigest()[:16]
    with commit_metadata(spark, metadata_json):
        spark.sql(f"ALTER TABLE {q(output)} SET TBLPROPERTIES ('{SYSTEM_PREFIX}heartbeat' = '{beat}')")


def write_bootstrap(spark: SparkSession, output: str, df: DataFrame, metadata_json: str) -> None:
    """Create the output table (CDF on from birth) with its first watermarks, in one commit."""
    check_no_system_columns(df, context=f"bootstrap of {output}")
    with commit_metadata(spark, metadata_json):
        (
            df.write.format("delta")
            .option("delta.enableChangeDataFeed", "true")
            .saveAsTable(output)
        )


def current(spark: SparkSession, output: str) -> DataFrame:
    return spark.table(output)
