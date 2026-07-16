"""Delta table plumbing: identifiers, version pins, and watermarks in commit metadata.

The watermark scheme is the load-bearing design decision (see ``docs/design.md``): the map
``{source → last processed Delta version}`` is stored as **custom commit metadata on the output's own
write commit**, so the data advance and the watermark advance are one atomic Delta commit — exactly-once
by construction, with no control table and no second source of truth. Recovery reads the output's
history for the latest trickle-stamped commit.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, SparkSession

# The key under which the watermark map travels in a commit's userMetadata JSON.
METADATA_KEY = "trickle_spark"

# Session conf Delta reads the commit metadata from. Session-global: fine under the single-writer-per-
# output assumption this library makes everywhere; `commit_metadata` scopes it to one operation.
_USER_METADATA_CONF = "spark.databricks.delta.commitInfo.userMetadata"


def q(name: str) -> str:
    """Backtick-quote a (possibly multi-part) table name. Literal dots inside a part are unsupported."""
    return ".".join(f"`{p}`" for p in name.split("."))


def table_exists(spark: SparkSession, name: str) -> bool:
    return spark.catalog.tableExists(name)


@dataclass(frozen=True)
class Pin:
    """A source resolved to a specific Delta version at plan time.

    Reads happen *as of* the pin, and the pin is what gets recorded — concurrent producer commits
    mid-run land cleanly in the next window. ``table_id`` guards against a dropped-and-recreated
    source, whose version counter resets: an id mismatch means the watermark is meaningless and the
    source must be read comprehensively, never as a (silently wrong) window.
    """

    version: int
    table_id: str


def pin_table(spark: SparkSession, name: str) -> Pin:
    dt = DeltaTable.forName(spark, name)
    table_id = dt.detail().select("id").collect()[0][0]
    version = dt.history(1).select("version").collect()[0][0]
    return Pin(version=int(version), table_id=str(table_id))


def read_as_of(spark: SparkSession, name: str, version: int) -> DataFrame:
    return spark.read.format("delta").option("versionAsOf", version).table(name)


def commit_metadata_json(kind: str, pins: dict[str, Pin], *, tag: str | None = None) -> str:
    payload = {
        "kind": kind,
        "sources": {s: {"version": p.version, "table_id": p.table_id} for s, p in sorted(pins.items())},
    }
    if tag is not None:
        payload["tag"] = tag  # an opaque caller token (e.g. an external epoch) — see read_tag
    return json.dumps({METADATA_KEY: payload})


@contextmanager
def commit_metadata(spark: SparkSession, metadata_json: str):
    """Attach userMetadata to the Delta commit(s) made inside the block."""
    prior = spark.conf.get(_USER_METADATA_CONF, None)
    spark.conf.set(_USER_METADATA_CONF, metadata_json)
    try:
        yield
    finally:
        if prior is None:
            spark.conf.unset(_USER_METADATA_CONF)
        else:
            spark.conf.set(_USER_METADATA_CONF, prior)


def _latest_payload(spark: SparkSession, output: str) -> dict | None:
    """The parsed trickle payload of the output's latest trickle-stamped commit, or ``None``."""
    if not table_exists(spark, output):
        return None
    hist = (
        DeltaTable.forName(spark, output)
        .history()
        .select("version", "userMetadata")
        .where(f"userMetadata LIKE '%{METADATA_KEY}%'")
        .orderBy("version", ascending=False)
        .limit(1)
        .collect()
    )
    if not hist:
        return None
    try:
        return json.loads(hist[0].userMetadata)[METADATA_KEY]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def read_tag(spark: SparkSession, table: str) -> str | None:
    """The opaque ``tag`` of ``table``'s latest trickle-stamped commit (see
    :func:`commit_metadata_json`) — the replay guard for :func:`~.bridge.apply_changes`: a caller
    stamping each landing with its own epoch can tell whether that epoch already landed."""
    payload = _latest_payload(spark, table)
    return payload.get("tag") if payload else None


def read_watermarks(spark: SparkSession, output: str) -> dict[str, Pin] | None:
    """Recover the watermark map from the output's latest trickle-stamped commit.

    ``None`` means no usable watermark: the table doesn't exist (bootstrap), was never written by
    trickle-spark, or its trickle commits have aged out of the Delta log (`delta.logRetentionDuration`)
    — all of which degrade to the same comprehensive fallback, so they need no distinction here.
    """
    payload = _latest_payload(spark, output)
    if payload is None:
        return None
    return {
        source: Pin(version=int(entry["version"]), table_id=str(entry["table_id"]))
        for source, entry in payload.get("sources", {}).items()
    }
