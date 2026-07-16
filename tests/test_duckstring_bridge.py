"""The worked hybrid example: a Duckstring (DuckDB) Trickle pipeline with one Spark-sized stage —
trickle-spark speaking duckstring's published format **natively**, duckstring unchanged.

The scenario from the design conversation: 90% of a pipeline is DuckDB-sized, one join is not. The
"Spark Pond" in the middle reads the upstream Ponds' published parquet directly and hands its change
back at each epoch, with **no landing tables and no bookkeeping state anywhere**:

  duckdb orders/catalog ──(published parts, read by ts.duckstring_source)──▶ Spark plan, tagged f
  duckdb priced  ◀──(ts.changes_at_tag(f) → duckstring apply_zset at f)──── Delta priced

Inbound, the watermark is duckstring's epoch ``f`` riding the Spark output's own commit metadata like
any Delta pin. Outbound, the change to hand back *at* an epoch is the CDF of the commits tagged with
it — content-addressed, so a duckstring crash-replay at the same ``f`` re-lands the identical window
(and the Spark run itself skips, its pins unchanged). This drives duckstring's ``trickle`` package on
a bare DuckDB connection with an explicit data-plane export (its host seam — no Catchment needed); in
a real deployment the loop below is one Pond's ripple, ``pond.f`` supplying the epochs. Skipped when
duckstring isn't installed (``pip install duckstring`` or ``-e`` the repo to exercise it)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("duckstring.trickle")

import duckdb  # noqa: E402 - guarded by the importorskip (duckdb arrives with duckstring)
from duckstring.trickle import io as dtio  # noqa: E402

import trickle_spark as ts  # noqa: E402
from trickle_spark import D_COL  # noqa: E402
from trickle_spark.duckstring import DUCK_D  # noqa: E402


def epoch(n: int) -> datetime:
    return datetime(2026, 1, n, tzinfo=timezone.utc)


@pytest.fixture
def dataplane(monkeypatch):
    monkeypatch.setenv("DUCKSTRING_DATA_PLANE", "parquet")  # the flat published layout the reader speaks
    from duckstring.dataplane import get_data_plane

    return get_data_plane()


def merge_and_publish(dp, con, table, sql, f, pk, data_dir) -> None:
    """A duckstring Pond run in miniature: merge the complete state at epoch ``f``, publish the plane."""
    dtio.merge_table(con, table, con.sql(sql), f, pk)
    dp.export(con, data_dir, mode="overwrite", f=f)


def emit_at(spark, con, output, duck_table, pk, f) -> None:
    """The Spark Pond's emit ripple: the change the run tagged ``f`` applied, landed in the duckdb
    Trickle at ``f``. No consumer watermark anywhere — the window is content-addressed by the tag."""
    d = ts.changes_at_tag(spark, output, f.isoformat())
    pdf = d.zset.toPandas().rename(columns={D_COL: DUCK_D})
    con.register("_emit_z", pdf)
    dtio.apply_zset(con, duck_table, con.sql("SELECT * FROM _emit_z"), f, pk)
    con.unregister("_emit_z")


def duck_state(con, table, cols) -> list:
    dtio.reconstruct_current(con, table).create_view("_cur", replace=True)
    sel = ", ".join(f'"{c}"' for c in cols)
    return sorted(tuple(r) for r in con.sql(f"SELECT {sel} FROM _cur").fetchall())


def test_duckstring_spark_round_trip_native(spark, db, dataplane, tmp_path):
    con = duckdb.connect()
    data_dir = tmp_path / "pond_data"
    orders_pk, catalog_pk, priced_pk = ("order_id",), ("product_id",), ("order_id",)
    priced = f"{db}.priced"
    plan = (
        ts.duckstring_source(data_dir, "orders", p=1.0).alias("o")
        .join(ts.duckstring_source(data_dir, "catalog", p=1.0).alias("c"), on="product_id")
        .mutate(amount="o.qty * c.price")
    )
    f1, f2, f3 = epoch(1), epoch(2), epoch(3)

    # ── run 1 (epoch f1): bootstrap end to end, straight off the published parquet ──
    merge_and_publish(dataplane, con, "orders",
                      "SELECT * FROM (VALUES (1, 'w', 2), (2, 'g', 1)) t(order_id, product_id, qty)",
                      f1, orders_pk, data_dir)
    merge_and_publish(dataplane, con, "catalog",
                      "SELECT * FROM (VALUES ('w', 10.0), ('g', 5.0)) t(product_id, price)",
                      f1, catalog_pk, data_dir)
    res = plan.merge_into(spark, priced, pk=priced_pk, tag=f1.isoformat())
    assert res.status == "bootstrap"
    emit_at(spark, con, priced, "priced", priced_pk, f1)
    assert duck_state(con, "priced", ("order_id", "amount")) == [(1, 20.0), (2, 5.0)]

    # ── run 2 (epoch f2): one catalog price changes; only affected rows cross each boundary ──
    merge_and_publish(dataplane, con, "catalog",
                      "SELECT * FROM (VALUES ('w', 12.0), ('g', 5.0)) t(product_id, price)",
                      f2, catalog_pk, data_dir)
    res = plan.merge_into(spark, priced, pk=priced_pk, tag=f2.isoformat())
    assert res.status == "incremental"
    emit_at(spark, con, priced, "priced", priced_pk, f2)
    assert duck_state(con, "priced", ("order_id", "amount")) == [(1, 24.0), (2, 5.0)]
    # the changelog window a downstream duckdb consumer reads is just the touched order
    downstream = dtio.read_registry_delta(con, "priced", f1, f2, priced_pk)
    assert not downstream.is_full
    downstream.zset.create_view("_win", replace=True)
    assert {r[0] for r in con.sql("SELECT order_id FROM _win").fetchall()} == {1}

    # a duckstring replay at f2 is exactly-once end to end: the Spark run skips (pins unchanged)...
    assert plan.merge_into(spark, priced, pk=priced_pk, tag=f2.isoformat()).status == "skipped"
    # ...and the re-emit recovers the identical tagged window, rewriting the same epoch idempotently
    emit_at(spark, con, priced, "priced", priced_pk, f2)
    assert duck_state(con, "priced", ("order_id", "amount")) == [(1, 24.0), (2, 5.0)]

    # ── run 3 (epoch f3): an order deleted upstream propagates all the way through ──
    merge_and_publish(dataplane, con, "orders",
                      "SELECT * FROM (VALUES (1, 'w', 2)) t(order_id, product_id, qty)",
                      f3, orders_pk, data_dir)
    res = plan.merge_into(spark, priced, pk=priced_pk, tag=f3.isoformat())
    assert res.status == "incremental"
    emit_at(spark, con, priced, "priced", priced_pk, f3)
    assert duck_state(con, "priced", ("order_id", "amount")) == [(1, 24.0)]

    # parity: the duckdb reconstruction and the Spark table agree exactly
    spark_rows = sorted(tuple(r) for r in spark.table(priced).select("order_id", "amount").collect())
    assert duck_state(con, "priced", ("order_id", "amount")) == spark_rows


def test_duckstring_source_full_read_after_refresh(spark, db, dataplane, tmp_path):
    """A raised floor (a duckstring refresh / retention trim) must read as full, never a wrong window."""
    con = duckdb.connect()
    data_dir = tmp_path / "pond_data"
    f1, f2 = epoch(1), epoch(2)
    merge_and_publish(dataplane, con, "items", "SELECT * FROM (VALUES (1, 'a')) t(item_id, v)",
                      f1, ("item_id",), data_dir)
    out = f"{db}.mirror"
    plan = ts.duckstring_source(data_dir, "items", p=1.0)
    plan.merge_into(spark, out, pk=("item_id",), tag=f1.isoformat())

    # wipe and re-bootstrap the source (duckstring refresh semantics: the floor jumps to the new f)
    dtio.drop_table(con, "items")
    merge_and_publish(dataplane, con, "items", "SELECT * FROM (VALUES (1, 'a2'), (2, 'b')) t(item_id, v)",
                      f2, ("item_id",), data_dir)
    res = plan.merge_into(spark, out, pk=("item_id",), tag=f2.isoformat())
    assert res.status == "comprehensive"  # floor(f2) > watermark(f1) → full read, diffed
    assert sorted((r.item_id, r.v) for r in spark.table(out).collect()) == [(1, "a2"), (2, "b")]