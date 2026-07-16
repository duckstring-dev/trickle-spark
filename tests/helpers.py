"""Shared test helpers: CDF-enabled source tables + assertions over a table's own change feed."""

from __future__ import annotations

from delta.tables import DeltaTable


def make_source(spark, name, rows, schema="id INT, v STRING"):
    df = spark.createDataFrame(rows, schema)
    df.write.format("delta").option("delta.enableChangeDataFeed", "true").saveAsTable(name)


def latest_version(spark, name) -> int:
    return DeltaTable.forName(spark, name).history(1).collect()[0].version


def cdf_at(spark, name, version):
    return (
        spark.read.format("delta")
        .option("readChangeFeed", "true")
        .option("startingVersion", version)
        .option("endingVersion", version)
        .table(name)
        .collect()
    )


def table_rows(spark, name, *cols):
    return sorted(tuple(getattr(r, c) for c in cols) for r in spark.table(name).collect())
