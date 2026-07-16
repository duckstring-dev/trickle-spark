"""The worked hybrid example: a Duckstring (DuckDB) Trickle pipeline with one Spark-sized stage.

The scenario from the design conversation — 90% of a pipeline is DuckDB-sized, one join is not.
A "Spark Pond" sits in the middle: its ripples land the upstream duckstring deltas on Delta tables,
run a trickle-spark plan over them, and emit the output's CDF window back into the duckstring world.
Neither engine learns the other's clock: duckstring windows by its epoch ``f``, Delta by its table
version, and the crossing is always a full-row Z-set.

  duckdb orders/catalog ──(read_registry_delta → apply_changes)──▶ Delta landing tables
                                                                        │  ts.source(...).join(...)
  duckdb priced  ◀──(table_changes → duckstring apply_zset at f)── Delta priced

This test drives the whole loop with duckstring's ``trickle`` package directly on a bare DuckDB
connection (its host seam — no Catchment/Pond runtime needed). In a real deployment the two bridge
functions below are the bodies of a Pond's ripples: ``pond.f`` supplies the epochs, and the consumed
Spark pin is persisted in the Pond's registry keyed by ``f`` (as ``_bridge_pin`` is here), which makes
the outbound window **content-addressed per run** — a crash replay at the same ``f`` re-lands the same
change, and ``apply_changes``' tag makes the inbound landing a no-op on replay. Skipped when
duckstring isn't installed; run ``pip install duckstring`` (or ``-e`` the repo) to exercise it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("duckstring.trickle")

import duckdb  # noqa: E402 - guarded by the importorskip (duckdb arrives with duckstring)
from duckstring.trickle import io as dtio  # noqa: E402
from duckstring.trickle.context import NEVER  # noqa: E402
from duckstring.trickle.context import SYSTEM_PREFIX as DUCK_PREFIX

import trickle_spark as ts  # noqa: E402
from trickle_spark import D_COL  # noqa: E402

DUCK_D = f"{DUCK_PREFIX}d"


def epoch(n: int) -> datetime:
    return datetime(2026, 1, n, tzinfo=timezone.utc)


# ─── the two bridging ripples (the bodies a Spark Pond's ripples would have) ─────


def land_into_spark(spark, con, table, landing, pk, previous_f, f) -> bool:
    """INBOUND: a duckstring source's Z-set window ``(previous_f, f]`` → the Delta landing table.
    ``f`` rides as the idempotency tag, so a duckstring replay at the same epoch lands nothing."""
    d = dtio.read_registry_delta(con, table, previous_f, f, pk)
    pdf = d.zset.df().rename(columns={DUCK_D: D_COL})
    if pdf.empty and not d.is_full:
        return False  # nothing in the window — in a real Pond the run wouldn't have fired at all
    return ts.apply_changes(spark, landing, spark.createDataFrame(pdf), pk=pk, tag=f.isoformat(), full=d.is_full)


def emit_from_spark(spark, con, output, duck_table, pk, f) -> None:
    """OUTBOUND: the Spark output's change since the last consumed pin → the duckdb Trickle table,
    stamped at this run's ``f``. The consumed pin is persisted per ``f`` in the registry
    (``_bridge_pin``), so the window is content-addressed: a replay at ``f`` re-reads from the same
    prior pin and duckstring's ``apply_zset`` rewrites the same epoch window idempotently."""
    con.execute(
        "CREATE TABLE IF NOT EXISTS _bridge_pin (f TIMESTAMPTZ, version BIGINT, table_id VARCHAR)"
    )
    con.execute("DELETE FROM _bridge_pin WHERE f >= ?", [f])  # a replay re-derives from the prior pin
    prior = con.execute("SELECT version, table_id FROM _bridge_pin ORDER BY f DESC LIMIT 1").fetchone()
    after = ts.Pin(version=prior[0], table_id=prior[1]) if prior else None

    d, pin = ts.table_changes(spark, output, after)
    pdf = d.zset.toPandas().rename(columns={D_COL: DUCK_D})
    con.register("_bridge_z", pdf)
    if d.is_full:  # bootstrap / recreated output / expired window → hand duckstring the clean state
        state = con.sql(f"SELECT * EXCLUDE ({DUCK_D}) FROM _bridge_z WHERE {DUCK_D} > 0")
        dtio.merge_table(con, duck_table, state, f, pk)
    else:
        dtio.apply_zset(con, duck_table, con.sql("SELECT * FROM _bridge_z"), f, pk)
    con.execute("INSERT INTO _bridge_pin VALUES (?, ?, ?)", [f, pin.version, pin.table_id])
    con.unregister("_bridge_z")


# ─── the pipeline ────────────────────────────────────────────────────────────────


def duck_state(con, table, cols) -> list:
    dtio.reconstruct_current(con, table).create_view("_bridge_cur", replace=True)
    sel = ", ".join(f'"{c}"' for c in cols)
    return sorted(tuple(r) for r in con.sql(f"SELECT {sel} FROM _bridge_cur").fetchall())


def merge_duck(con, table, sql, f, pk) -> None:
    dtio.merge_table(con, table, con.sql(sql), f, pk)


def test_duckstring_spark_round_trip(spark, db):
    con = duckdb.connect()
    orders_pk, catalog_pk, priced_pk = ("order_id",), ("product_id",), ("order_id",)
    landing_o, landing_c, priced = f"{db}.orders", f"{db}.catalog", f"{db}.priced"
    plan = (
        ts.source(landing_o, p=1.0).alias("o")
        .join(ts.source(landing_c, p=1.0).alias("c"), on="product_id")
        .mutate(amount="o.qty * c.price")
    )
    f1, f2, f3 = epoch(1), epoch(2), epoch(3)

    # ── run 1 (epoch f1): bootstrap end to end ──
    merge_duck(con, "orders", "SELECT * FROM (VALUES (1, 'w', 2), (2, 'g', 1)) t(order_id, product_id, qty)",
               f1, orders_pk)
    merge_duck(con, "catalog", "SELECT * FROM (VALUES ('w', 10.0), ('g', 5.0)) t(product_id, price)",
               f1, catalog_pk)
    assert land_into_spark(spark, con, "orders", landing_o, orders_pk, NEVER, f1)
    assert land_into_spark(spark, con, "catalog", landing_c, catalog_pk, NEVER, f1)
    assert plan.merge_into(spark, priced, pk=priced_pk).status == "bootstrap"
    emit_from_spark(spark, con, priced, "priced", priced_pk, f1)
    assert duck_state(con, "priced", ("order_id", "amount")) == [(1, 20.0), (2, 5.0)]

    # ── run 2 (epoch f2): one catalog price changes; only the affected rows cross each boundary ──
    merge_duck(con, "catalog", "SELECT * FROM (VALUES ('w', 12.0), ('g', 5.0)) t(product_id, price)",
               f2, catalog_pk)
    assert land_into_spark(spark, con, "catalog", landing_c, catalog_pk, f1, f2)
    assert not land_into_spark(spark, con, "orders", landing_o, orders_pk, f1, f2)  # empty window
    assert plan.merge_into(spark, priced, pk=priced_pk).status == "incremental"
    emit_from_spark(spark, con, priced, "priced", priced_pk, f2)
    assert duck_state(con, "priced", ("order_id", "amount")) == [(1, 24.0), (2, 5.0)]
    # the duckdb changelog window a downstream duckdb consumer would read is just the touched order
    downstream = dtio.read_registry_delta(con, "priced", f1, f2, priced_pk)
    assert not downstream.is_full
    downstream.zset.create_view("_bridge_win", replace=True)
    assert {r[0] for r in con.sql("SELECT order_id FROM _bridge_win").fetchall()} == {1}

    # a duckstring replay at f2 is exactly-once on both sides: the landing tag skips it...
    assert not land_into_spark(spark, con, "catalog", landing_c, catalog_pk, f1, f2)
    # ...and re-emitting at f2 rewrites the same epoch window (no phantom changes downstream)
    emit_from_spark(spark, con, priced, "priced", priced_pk, f2)
    assert duck_state(con, "priced", ("order_id", "amount")) == [(1, 24.0), (2, 5.0)]

    # ── run 3 (epoch f3): an order is deleted upstream; the retraction propagates all the way ──
    merge_duck(con, "orders", "SELECT * FROM (VALUES (1, 'w', 2)) t(order_id, product_id, qty)",
               f3, orders_pk)
    assert land_into_spark(spark, con, "orders", landing_o, orders_pk, f2, f3)
    assert plan.merge_into(spark, priced, pk=priced_pk).status == "incremental"
    emit_from_spark(spark, con, priced, "priced", priced_pk, f3)
    assert duck_state(con, "priced", ("order_id", "amount")) == [(1, 24.0)]

    # parity: the duckdb reconstruction and the Spark table agree exactly
    spark_rows = sorted(tuple(r) for r in spark.table(priced).select("order_id", "amount").collect())
    assert duck_state(con, "priced", ("order_id", "amount")) == spark_rows
